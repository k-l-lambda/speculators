from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from speculators import SpeculatorModelConfig

__all__ = [
    "MTPSpeculatorConfig",
]


@SpeculatorModelConfig.register("mtp")
class MTPSpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration for Multi-Token Prediction (MTP) speculator.

    MTP uses the same vocabulary as the verifier model (no vocab mapping).
    The architecture consists of: embed_tokens + enorm + hnorm + eh_proj +
    a frozen/trainable decoder layer + shared_head.

    :param decoder_layer_config: Configuration for the MTP decoder layer
    :param freeze_decoder: Whether to freeze the decoder layer during training
    :param verifier_layer_idx: Which verifier layer to copy for decoder init
    """

    speculators_model_type: Literal["mtp"] = "mtp"
    architectures: list[str] = Field(
        default_factory=lambda: ["MTPSpeculator"],
        description="Model architectures that can load these weights",
    )

    decoder_layer_config: PretrainedConfig = Field(
        default_factory=LlamaConfig,
        description="Configuration for the MTP decoder layer",
    )

    freeze_decoder: bool = Field(
        default=True,
        description="Whether to freeze the decoder layer during training",
    )

    verifier_layer_idx: int = Field(
        default=-1,
        description="Verifier layer index to copy for decoder init (-1 = last layer)",
    )

    @field_serializer("decoder_layer_config")
    def serialize_decoder_config(self, value: PretrainedConfig) -> dict:
        return value.to_diff_dict()

    @field_validator("decoder_layer_config", mode="before")
    @classmethod
    def validate_decoder_config(cls, value: Any) -> PretrainedConfig:
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = LlamaConfig
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value
