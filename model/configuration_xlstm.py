import json
from typing import Any, Dict, Optional

from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf
from transformers.configuration_utils import PretrainedConfig
from xlstm import xLSTMLMModelConfig

# from .config_presets import xlstm_cfg_map


class xLSTMConfig(PretrainedConfig):
    """XLSTM configuration class.
    We seperate the specific xLSTM model configuration
    from the rest due to the heavy nesting of the configuration.
    """

    model_type = "xlstm"

    def __init__(
        self, vocab_size: int = 32000, config: Optional[Dict[str, Any]] = None, **kwargs
    ):
        super().__init__(**kwargs)

        cfg = OmegaConf.create(config)
        cfg["vocab_size"] = vocab_size
        for key, value in kwargs.items():
            cfg[key] = value

        self._xlstm_config = cfg
        self.vocab_size = vocab_size
        self.embedding_dim = cfg.get("embedding_dim")
        self.context_length = cfg.get("context_length")

    def to_xlstm_config(self):
        return from_dict(
            data_class=xLSTMLMModelConfig,
            data=OmegaConf.to_container(self._xlstm_config),
            config=DaciteConfig(strict=True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the configuration to a dictionary for serialization.
        """
        output = super().to_dict()
        output["_xlstm_config"] = OmegaConf.to_container(
            self._xlstm_config, resolve=True
        )
        relevant_keys = [
            "vocab_size",
            "embedding_dim",
            "context_length",
            "torch_dtype",
            "_xlstm_config",
            "transformers_version",
            "architectures",
            "model_type",
        ]
        output_ = output.copy()
        for key in output.keys():
            if key not in relevant_keys:
                output_.pop(key)
        return output_

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        """
        Creates a configuration instance from a dictionary.
        """
        xlstm_config = config_dict.pop("_xlstm_config")
        vocab_size = config_dict.pop("vocab_size")
        config = cls(vocab_size=vocab_size, config=xlstm_config)
        if "auto_map" in config_dict and config_dict["auto_map"]:
            setattr(config, "auto_map", config_dict.pop("auto_map"))

        # breakpoint()
        # config.xlstm_config = xlstm_config
        if "return_unused_kwargs" in kwargs and kwargs["return_unused_kwargs"]:
            return config, {}

        return config

    def to_json_string(self, *args, **kwargs) -> str:
        """
        Serializes the instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json_string(cls, json_string: str):
        """
        Deserializes the instance from a JSON string.
        """
        config_dict = json.loads(json_string)
        return cls.from_dict(config_dict)
