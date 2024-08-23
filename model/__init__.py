from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

from .configuration_xlstm import xLSTMConfig
from .modeling_xlstm import xLSTMForCausalLM

AutoConfig.register("xlstm", xLSTMConfig)
AutoModelForCausalLM.register(xLSTMConfig, xLSTMForCausalLM)
