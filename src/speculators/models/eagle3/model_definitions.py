# ruff: noqa: ERA001
from typing import Any

import torch
from transformers import Cache, LlamaConfig, PretrainedConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs

from speculators.models import base_components


class Eagle3FirstLayerMixin:
    """Shared Eagle3 first-layer modifications for any decoder layer.

    Patches q/k/v projections to accept 2x hidden_size input (cat([embeds, hidden]))
    and overrides forward to split, normalize, and recombine before attention.
    """

    # Provided by the decoder layer base class
    self_attn: Any
    input_layernorm: Any
    post_attention_layernorm: Any
    mlp: Any
    norm_before_residual: bool
    hidden_norm: Any

    def _patch_eagle3_projections(
        self,
        config: PretrainedConfig,
        norm_class: type[torch.nn.Module],
        norm_before_residual: bool,
    ):
        """Replace q/k/v projections with 2x hidden_size input and add hidden_norm."""
        self.norm_before_residual = norm_before_residual
        self.hidden_norm = norm_class(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn.q_proj = torch.nn.Linear(
            2 * config.hidden_size,  # previous: config.hidden_size
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.self_attn.k_proj = torch.nn.Linear(
            2 * config.hidden_size,  # previous: config.hidden_size
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.self_attn.v_proj = torch.nn.Linear(
            2 * config.hidden_size,  # previous: config.hidden_size
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],  # type: ignore[valid-type]
    ) -> torch.Tensor:
        # Previously in the parent DecoderLayer:
        #   residual = hidden_states
        #   hidden_states = self.input_layernorm(hidden_states)

        # ##### Start of Eagle3 modifications #####

        # hidden_states are cat([embeds, hidden], dim=-1)
        # so residual should be hidden part only, and embeds should be normalized
        mid = hidden_states.shape[2] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        residual = hidden

        # Apply norms
        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        if self.norm_before_residual:
            residual = hidden  # set residual to normalized hidden
        hidden_states = torch.cat([embeds, hidden], dim=-1)
        if torch.__version__ >= "2.10":
            # As of `torch==2.10`, compile attempts to fuse together too many
            # ops, resulting in a fused kernel that exceeds shared memory limits
            # For now, we force a graph break to prevent this
            # https://github.com/pytorch/pytorch/issues/175250
            torch._dynamo.graph_break()  # noqa: SLF001

        # ##### End of Eagle3 modifications #####

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states  # noqa: RET504


class LlamaDecoderEagle3FirstLayer(Eagle3FirstLayerMixin, LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
        norm_before_residual: bool = False,
    ):
        super().__init__(config, layer_idx)
        self._patch_eagle3_projections(config, LlamaRMSNorm, norm_before_residual)


class Qwen3DecoderEagle3FirstLayer(Eagle3FirstLayerMixin, Qwen3DecoderLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        norm_before_residual: bool = False,
    ):
        super().__init__(config, layer_idx)
        self._patch_eagle3_projections(config, Qwen3RMSNorm, norm_before_residual)


# DeepSeek-V3 / Kimi-K2.5 Eagle3 first layer (MLA attention adaptation)
try:
    from speculators.models.base_components import (
        DeepseekV3DecoderLayer as _DsDecoderLayer,
        _BlockMaskCompatDecoder as _DsBlockMaskDecoder,
        DeepseekV3RMSNorm as _DsRMSNorm,
        DeepseekV3Config as _DsConfig,
        _HAS_DEEPSEEK,
    )

    if _HAS_DEEPSEEK:
        class DeepseekV3DecoderEagle3FirstLayer(Eagle3FirstLayerMixin, _DsBlockMaskDecoder):
            """Eagle3 first layer for K2.5/DeepSeek-V3 with MLA attention.

            MLA uses LoRA-style projections instead of standard q/k/v:
            - q_a_proj [H -> q_lora_rank] + q_b_proj [q_lora_rank -> heads*dim]
            - kv_a_proj_with_mqa [H -> kv_lora_rank+rope_dim] + kv_b_proj [...]

            We patch q_a_proj and kv_a_proj_with_mqa to accept 2*H input,
            leaving the b_proj (which generates actual Q/K/V) unchanged.
            """
            def __init__(
                self,
                config: _DsConfig,
                layer_idx: int,
                norm_before_residual: bool = False,
            ):
                # Set eager attention to avoid flash attention issues during training
                config._attn_implementation = "eager"
                super().__init__(config, layer_idx)
                self._patch_eagle3_projections_mla(config, _DsRMSNorm, norm_before_residual)

            def _patch_eagle3_projections_mla(self, config, norm_class, norm_before_residual):
                """Patch MLA input projections to accept 2*hidden_size."""
                self.norm_before_residual = norm_before_residual
                self.hidden_norm = norm_class(
                    config.hidden_size, eps=getattr(config, 'rms_norm_eps', 1e-5)
                )
                # MLA: patch LoRA input projections (a_proj) to accept 2*H
                self.self_attn.q_a_proj = torch.nn.Linear(
                    2 * config.hidden_size,
                    config.q_lora_rank,
                    bias=config.attention_bias,
                )
                self.self_attn.kv_a_proj_with_mqa = torch.nn.Linear(
                    2 * config.hidden_size,
                    config.kv_lora_rank + config.qk_rope_head_dim,
                    bias=config.attention_bias,
                )
                # q_b_proj, kv_b_proj, kv_a_layernorm, q_a_layernorm: unchanged

            def forward(self, hidden_states, attention_mask=None, position_ids=None,
                        past_key_values=None, use_cache=False, cache_position=None,
                        position_embeddings=None, **kwargs):
                # Convert Eagle3 BlockMask to standard 4D causal tensor for K2.5 MLA
                if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
                    try:
                        from torch.nn.attention.flex_attention import BlockMask
                        if isinstance(attention_mask, BlockMask):
                            seq_len = hidden_states.shape[1]
                            dtype = hidden_states.dtype
                            device = hidden_states.device
                            causal = torch.full(
                                (1, 1, seq_len, seq_len),
                                torch.finfo(dtype).min, dtype=dtype, device=device,
                            )
                            attention_mask = torch.triu(causal, diagonal=1)
                    except ImportError:
                        attention_mask = None

                # Eagle3 first-layer logic (reimplemented to handle K2.5 MLA 3-tuple return)
                mid = hidden_states.shape[2] // 2
                embeds, hidden = hidden_states.split(mid, dim=-1)
                residual = hidden
                embeds = self.input_layernorm(embeds)
                hidden = self.hidden_norm(hidden)
                if self.norm_before_residual:
                    residual = hidden
                hidden_states = torch.cat([embeds, hidden], dim=-1)

                # K2.5 attention returns (output, weights, past_kv) — 3 values
                attn_result = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                attn_output = attn_result[0] if isinstance(attn_result, tuple) else attn_result
                hidden_states = residual + attn_output

                residual = hidden_states
                hidden_states = self.post_attention_layernorm(hidden_states)
                hidden_states = self.mlp(hidden_states)
                hidden_states = residual + hidden_states
                return hidden_states
except ImportError:
    _HAS_DEEPSEEK = False


model_classes: dict[str, base_components.ModelComponents] = {
    "llama": base_components.override_components(
        "llama", first_layer_class=LlamaDecoderEagle3FirstLayer
    ),
    "qwen3": base_components.override_components(
        "qwen3", first_layer_class=Qwen3DecoderEagle3FirstLayer
    ),
}

# Register kimi_k2 Eagle3 first layer if available
if _HAS_DEEPSEEK:
    _kimi_eagle3 = base_components.override_components(
        "kimi_k2", first_layer_class=DeepseekV3DecoderEagle3FirstLayer
    )
    model_classes["kimi_k2"] = _kimi_eagle3
    model_classes["deepseek_v3"] = _kimi_eagle3  # same config type from DeepseekV3Config
