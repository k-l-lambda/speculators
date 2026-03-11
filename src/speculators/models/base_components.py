"""Shared base model components for all speculator types."""

from typing import NamedTuple

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

# DeepSeek-V3 / Kimi-K2.5: import from bundled k2_mtp_config
# These use MLA attention (not standard q/k/v) and MoE
import os as _os
from pathlib import Path as _Path

_k2_config_dir = _Path(__file__).parent.parent.parent.parent / "scripts" / "k2_mtp_config"
if _k2_config_dir.exists():
    import importlib.util as _ilu
    _spec_cfg = _ilu.spec_from_file_location(
        "configuration_deepseek", _k2_config_dir / "configuration_deepseek.py"
    )
    _mod_cfg = _ilu.module_from_spec(_spec_cfg)
    _spec_cfg.loader.exec_module(_mod_cfg)

    # Patch sys.modules so modeling_deepseek can find configuration_deepseek
    import sys as _sys
    _sys.modules["configuration_deepseek"] = _mod_cfg

    _spec_mod = _ilu.spec_from_file_location(
        "modeling_deepseek", _k2_config_dir / "modeling_deepseek.py"
    )
    _mod_mod = _ilu.module_from_spec(_spec_mod)
    _spec_mod.loader.exec_module(_mod_mod)
    _sys.modules["modeling_deepseek"] = _mod_mod

    DeepseekV3DecoderLayer = _mod_mod.DeepseekV3DecoderLayer
    DeepseekV3RMSNorm = _mod_mod.DeepseekV3RMSNorm
    DeepseekV3Config = _mod_cfg.DeepseekV3Config
    _HAS_DEEPSEEK = True
else:
    _HAS_DEEPSEEK = False


class ModelComponents(NamedTuple):
    """Container for the components of a speculators model.

    This groups the building blocks needed to construct a model, enabling
    architecture-agnostic code and selective component overriding for
    speculative decoding algorithms.

    Attributes:
        first_layer_class: Class for the first decoder layer. Can be customized
            for speculative decoding while keeping other layers standard.
        decoder_layer_class: Class for standard decoder layers used throughout
            the rest of the model.
        norm_class: Normalization layer class (e.g., LlamaRMSNorm, Qwen3RMSNorm).
        rotary_emb_class: Rotary positional embedding class for the model.
    """

    first_layer_class: type
    decoder_layer_class: type
    norm_class: type
    rotary_emb_class: type


model_classes: dict[str, ModelComponents] = {
    "llama": ModelComponents(
        LlamaDecoderLayer,  # first_layer_class (same as decoder for base models)
        LlamaDecoderLayer,
        LlamaRMSNorm,
        LlamaRotaryEmbedding,
    ),
    "qwen3": ModelComponents(
        Qwen3DecoderLayer,  # first_layer_class (same as decoder for base models)
        Qwen3DecoderLayer,
        Qwen3RMSNorm,
        Qwen3RotaryEmbedding,
    ),
}

# Conditionally register kimi_k2 if DeepSeek modeling is available
if _HAS_DEEPSEEK:
    import torch as _torch

    class _DeepseekRotaryProxy:
        def __init__(self, config): pass
        def __call__(self, hidden_states, position_ids): return (None, None)

    class _BlockMaskCompatDecoder(DeepseekV3DecoderLayer):
        def forward(self, hidden_states, attention_mask=None, position_ids=None,
                    position_embeddings=None, **kwargs):
            if attention_mask is not None and not isinstance(attention_mask, _torch.Tensor):
                try:
                    from torch.nn.attention.flex_attention import BlockMask
                    if isinstance(attention_mask, BlockMask):
                        seq_len = hidden_states.shape[1]
                        causal = _torch.full(
                            (1, 1, seq_len, seq_len),
                            _torch.finfo(hidden_states.dtype).min,
                            dtype=hidden_states.dtype, device=hidden_states.device,
                        )
                        attention_mask = _torch.triu(causal, diagonal=1)
                except ImportError:
                    attention_mask = None
            # Eagle3 uses 1-indexed position_ids, but K2.5 rotary embedding
            # returns cos[:seq_len] (0-indexed). Shift to avoid OOB at seq_len boundary.
            if position_ids is not None and position_ids.min() >= 1:
                position_ids = position_ids - 1
            # Reset cache_position if passed: Eagle3 TTT uses arange(step*S, (step+1)*S)
            # but K2.5 attention interprets large cache_position as kv_seq_len offset,
            # causing attention mask size mismatch in deeper TTT steps.
            if 'cache_position' in kwargs and kwargs['cache_position'] is not None:
                kwargs['cache_position'] = _torch.arange(
                    hidden_states.shape[1], device=hidden_states.device
                )
            return super().forward(
                hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, position_embeddings=position_embeddings,
                **kwargs
            )

    _deepseek_components = ModelComponents(
        _BlockMaskCompatDecoder,
        _BlockMaskCompatDecoder,
        DeepseekV3RMSNorm,
        _DeepseekRotaryProxy,
    )
    model_classes["kimi_k2"] = _deepseek_components
    model_classes["deepseek_v3"] = _deepseek_components


def override_components(model_type: str, **overrides) -> ModelComponents:
    """Override specific components from a base model architecture.

    Used for speculative decoding to swap custom layers (typically first_layer_class)
    while inheriting other components from the base model.

    Args:
        model_type: Base model type ("llama" or "qwen3").
        **overrides: Component fields to override (first_layer_class,
            decoder_layer_class, etc).

    Returns:
        ModelComponents with specified overrides applied.
    """
    base = model_classes[model_type]
    return base._replace(**overrides)
