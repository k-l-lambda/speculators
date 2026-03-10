"""
MTP (Multi-Token Prediction) Draft Model for speculators.

Architecture (DeepSeek-V3 / Kimi-K2.5 style):
    1. enorm(embed_tokens(x_{t+1}))   -- embed next ground-truth token
    2. hnorm(h_t)                       -- normalize verifier hidden state
    3. eh_proj(cat[embed, hidden])      -- fuse to hidden_size
    4. decoder_layer(fused)             -- standard decoder layer (frozen or trainable)
    5. shared_head(shared_head_norm(output)) -- predict token_{t+2}

Loss: cross-entropy against ground-truth token_{t+2}, or KL-divergence
against verifier logits computed from verifier hidden states.
"""

import warnings
from typing import ClassVar

import torch
import torch.nn as nn
from transformers import AutoConfig, PretrainedConfig

from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.model import SpeculatorModel
from speculators.models.mtp.config import MTPSpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from speculators.utils.loading import load_model_layers


def mtp_loss_ce(
    logits: torch.Tensor,       # [1, S, vocab_size]
    target_ids: torch.Tensor,   # [1, S]
    loss_mask: torch.Tensor | None,  # [1, S]
) -> torch.Tensor:
    """Cross-entropy loss for MTP: predict target_ids from logits."""
    logits_flat = logits.squeeze(0)   # [S, V]
    targets_flat = target_ids.squeeze(0)  # [S]
    ce = nn.functional.cross_entropy(logits_flat, targets_flat, reduction="none")
    if loss_mask is not None:
        mask = loss_mask.squeeze(0).float()
        return (ce * mask).sum() / (mask.sum() + 1e-5)
    return ce.mean()


def mtp_loss_kl(
    logits: torch.Tensor,       # [1, S, vocab_size]
    targets: torch.Tensor,      # [1, S, vocab_size] (verifier logits)
    loss_mask: torch.Tensor | None,  # [1, S]
) -> torch.Tensor:
    """KL-divergence loss: align draft distribution with verifier distribution."""
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    target_probs = nn.functional.softmax(targets, dim=-1)
    kl = nn.functional.kl_div(log_probs, target_probs, reduction="none")
    if loss_mask is not None:
        kl = kl * loss_mask.unsqueeze(-1)
        denom = loss_mask.sum(dim=1) + 1e-5
    else:
        denom = logits.shape[1]
    return (kl.sum(dim=(1, 2)) / denom).mean()


@torch.no_grad()
def mtp_accuracy(
    logits: torch.Tensor,       # [1, S, vocab_size]
    target_ids: torch.Tensor,   # [1, S]
    loss_mask: torch.Tensor | None,  # [1, S]
) -> torch.Tensor:
    """Top-1 accuracy for MTP predictions."""
    predicted = logits.argmax(dim=-1)
    correct = (predicted == target_ids)
    if loss_mask is not None:
        correct = correct & loss_mask.bool()
        return correct.float().sum() / (loss_mask.sum() + 1e-5)
    return correct.float().mean()


@SpeculatorModel.register("mtp")
class MTPDraftModel(SpeculatorModel):
    config_class: ClassVar[type[MTPSpeculatorConfig]] = MTPSpeculatorConfig

    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [
        "embed_tokens.weight",
        "verifier_norm.weight",
        "verifier_lm_head.weight",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    def __init__(self, config: MTPSpeculatorConfig):
        super().__init__(config=config)
        self.hidden_size = config.decoder_layer_config.hidden_size
        self.vocab_size = config.decoder_layer_config.vocab_size
        rms_norm_eps = getattr(config.decoder_layer_config, "rms_norm_eps", 1e-5)

        # MTP-specific layers
        self.enorm = nn.modules.normalization.RMSNorm(
            self.hidden_size, eps=rms_norm_eps,
        )
        self.hnorm = nn.modules.normalization.RMSNorm(
            self.hidden_size, eps=rms_norm_eps,
        )
        self.eh_proj = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)

        # Shared head (predicts next-next token, full vocab)
        self.shared_head_norm = nn.modules.normalization.RMSNorm(
            self.hidden_size, eps=rms_norm_eps,
        )
        self.shared_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

        # Load verifier components (embeddings, lm_head for KL loss, norm)
        self._setup_verifier_components(config)

        # Decoder layer(s): initialized by from_training_args or loaded
        # Use ModuleList for FSDP compat (verify_training_compatible checks .layers)
        self.layers = nn.ModuleList()

    def _setup_verifier_components(self, config: MTPSpeculatorConfig):
        """Load frozen verifier embeddings, lm_head, and norm for loss computation."""
        verifier_cfg = config.speculators_config.verifier
        if verifier_cfg.name_or_path is None:
            raise ValueError("VerifierConfig name_or_path is required.")

        verifier_model_config = AutoConfig.from_pretrained(verifier_cfg.name_or_path)
        if hasattr(verifier_model_config, "text_config"):
            verifier_model_config = verifier_model_config.text_config

        verifier_weights = load_model_layers(
            ["embed_tokens.weight", "lm_head.weight", "model.norm.weight"],
            verifier_cfg.name_or_path,
        )

        default_dtype = torch.bfloat16
        rms_norm_eps = getattr(verifier_model_config, "rms_norm_eps", 1e-5)

        # Embedding (frozen, used for input)
        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.hidden_size,
            padding_idx=getattr(verifier_model_config, "pad_token_id", None),
        )
        embed_w = verifier_weights["embed_tokens.weight"].to(default_dtype)
        self.embed_tokens.load_state_dict({"weight": embed_w})
        self.embed_tokens.weight.requires_grad = False

        # Verifier LM head (frozen, for KL-div loss targets)
        self.verifier_lm_head = nn.Linear(
            self.hidden_size, self.vocab_size, bias=False,
        )
        lm_head_w = verifier_weights.get("lm_head.weight", embed_w)
        self.verifier_lm_head.weight.data = lm_head_w.to(default_dtype).detach().clone()
        self.verifier_lm_head.weight.requires_grad = False

        # Verifier final norm (frozen, for KL-div loss targets)
        self.verifier_norm = nn.modules.normalization.RMSNorm(
            self.hidden_size, eps=rms_norm_eps,
        )
        if "model.norm.weight" in verifier_weights:
            norm_w = verifier_weights["model.norm.weight"].to(default_dtype)
            self.verifier_norm.load_state_dict({"weight": norm_w})
        self.verifier_norm.weight.requires_grad = False

        # Initialize shared_head from verifier lm_head
        self.shared_head.weight.data = lm_head_w.to(default_dtype).detach().clone()

    def forward(
        self,
        hidden_states: torch.Tensor,        # [1, S, hidden_size]
        input_ids: torch.Tensor,             # [1, S]
        lengths: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        verifier_last_hidden_states: torch.Tensor | None = None,
        loss_type: str = "ce",
        **kwargs,
    ):
        """MTP forward pass.

        After shift_batch alignment:
          - hidden_states[t] = verifier output at position t
          - input_ids[t] = token at position t+1
          - verifier_last_hidden_states[t] = verifier output at position t+1
        Target: predict token at position t+2 = input_ids[t+1]
        """
        device = hidden_states.device
        seq_len = hidden_states.shape[1]

        # Step 1: Embed input tokens and normalize
        with torch.no_grad():
            token_embed = self.embed_tokens(input_ids)
        embed_normed = self.enorm(token_embed)

        # Step 2: Normalize verifier hidden states
        hidden_normed = self.hnorm(hidden_states)

        # Step 3: Fuse
        fused = self.eh_proj(torch.cat([embed_normed, hidden_normed], dim=-1))

        # Step 4: Decoder layer
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        for layer in self.layers:
            layer_output = layer(fused, position_ids=position_ids)
            if isinstance(layer_output, tuple):
                fused = layer_output[0]
            else:
                fused = layer_output

        # Step 5: Predict
        logits = self.shared_head(self.shared_head_norm(fused))

        return_loss = verifier_last_hidden_states is not None
        if not return_loss:
            return logits

        # Target: input_ids shifted by 1 (predict t+2)
        target_ids = torch.cat([
            input_ids[:, 1:],
            input_ids.new_zeros(1, 1),
        ], dim=-1)

        # Mask out last position (no target)
        if loss_mask is not None:
            adjusted_mask = loss_mask.clone()
            adjusted_mask[:, -1] = 0
        else:
            adjusted_mask = torch.ones(1, seq_len, device=device)
            adjusted_mask[:, -1] = 0

        metrics = {}

        if loss_type == "kl":
            with torch.no_grad():
                verifier_logits = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
                verifier_targets = torch.cat([
                    verifier_logits[:, 1:, :],
                    verifier_logits.new_zeros(1, 1, verifier_logits.shape[-1]),
                ], dim=1)
            loss = mtp_loss_kl(logits, verifier_targets, adjusted_mask)
        else:
            loss = mtp_loss_ce(logits, target_ids, adjusted_mask)

        acc = mtp_accuracy(logits, target_ids, adjusted_mask)
        metrics["loss"] = loss.detach().clone()
        metrics["loss_0"] = loss.detach().clone()
        metrics["full_acc_0"] = acc

        return logits, loss, metrics

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        **kwargs,
    ) -> "MTPDraftModel":
        """Create MTP model from training arguments."""
        freeze_decoder = kwargs.get("freeze_decoder", True)
        verifier_layer_idx = kwargs.get("verifier_layer_idx", -1)

        config = MTPSpeculatorConfig(
            decoder_layer_config=verifier_config,
            freeze_decoder=freeze_decoder,
            verifier_layer_idx=verifier_layer_idx,
            speculators_config=SpeculatorsConfig(
                algorithm="mtp",
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=1),
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_config(
                    verifier_config,
                    name_or_path=kwargs["verifier_name_or_path"],
                ),
            ),
        )

        model = cls(config=config)
        model._init_decoder_from_verifier(
            kwargs["verifier_name_or_path"],
            verifier_config,
            verifier_layer_idx,
            freeze_decoder,
        )
        return model

    def _init_decoder_from_verifier(
        self,
        verifier_path: str,
        verifier_config: PretrainedConfig,
        layer_idx: int,
        freeze: bool,
    ):
        """Initialize the MTP decoder layer by copying weights from verifier."""
        import sys
        from pathlib import Path

        num_layers = verifier_config.num_hidden_layers
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx

        # Import DeepSeek modeling code
        config_dir = (
            Path(__file__).parent.parent.parent.parent
            / "scripts" / "k2_mtp_config"
        )
        if config_dir.exists():
            sys.path.insert(0, str(config_dir))
            try:
                from modeling_deepseek import DeepseekV3DecoderLayer
                from configuration_deepseek import DeepseekV3Config
            except ImportError:
                raise ImportError(
                    f"Cannot import DeepSeek modeling from {config_dir}. "
                    "Ensure modeling_deepseek.py and configuration_deepseek.py exist."
                )
            finally:
                sys.path.pop(0)
        else:
            raise FileNotFoundError(
                f"k2_mtp_config not found at {config_dir}"
            )

        ds_config = DeepseekV3Config(**verifier_config.to_dict())
        ds_config._attn_implementation = "eager"

        decoder_layer = DeepseekV3DecoderLayer(ds_config, layer_idx=layer_idx)
        self.layers.append(decoder_layer)

        # Load verifier weights for this layer
        layer_prefix = f"model.layers.{layer_idx}."
        try:
            weights = load_model_layers([f"{layer_prefix}*"], verifier_path)
            sd = {}
            for k, v in weights.items():
                short_key = k.removeprefix(layer_prefix)
                sd[short_key] = v.to(torch.bfloat16)
            missing, unexpected = decoder_layer.load_state_dict(sd, strict=False)
            if missing:
                warnings.warn(
                    f"Missing keys loading layer {layer_idx}: "
                    f"{missing[:5]}...",
                    UserWarning, stacklevel=2,
                )
        except Exception as e:
            warnings.warn(
                f"Could not load verifier layer {layer_idx}: {e}. "
                "Using random init.",
                UserWarning, stacklevel=2,
            )

        if freeze:
            for p in decoder_layer.parameters():
                p.requires_grad = False

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        train_kwargs = {"loss_type": kwargs.get("loss_type", "ce")}
        val_kwargs = {"loss_type": kwargs.get("loss_type", "ce")}
        return train_kwargs, val_kwargs
