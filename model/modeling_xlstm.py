from typing import Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from xlstm.components.init import small_init_init_
from xlstm.utils import WeightDecayOptimGroupMixin
from xlstm.xlstm_block_stack import xLSTMBlockStack as _xLSTMBlockStack

from .configuration_xlstm import xLSTMConfig


class xLSTMPreTrainedModel(PreTrainedModel):
    """Base class for all models."""

    config_class = xLSTMConfig


class xLSTMBlockStack(_xLSTMBlockStack):
    """Small wrapper to expose hidden states"""

    def forward(
        self, x: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Sequence[torch.Tensor]]:
        hidden_states = ()
        for block in self.blocks:
            x = block(x, **kwargs)
            hidden_states += (x,)

        x = self.post_blocks_norm(x)

        return x, hidden_states


class xLSTMModel(xLSTMPreTrainedModel):
    def __init__(self, config: xLSTMConfig):
        super().__init__(config)
        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim
        )
        _config = config.to_xlstm_config()

        self.emb_dropout = (
            nn.Dropout(_config.dropout)
            if _config.add_embedding_dropout
            else nn.Identity()
        )

        self.xlstm_block_stack = xLSTMBlockStack(config=_config)


    def forward(
        self,
        input_ids: torch.LongTensor,
        output_hidden_states: Optional[bool] = None,
        return_dict=Optional[bool],
    ) -> Union[Tuple, BaseModelOutput]:
        token_embedding = self.token_embedding(input_ids)
        x = self.emb_dropout(token_embedding)
        x, hidden_states = self.xlstm_block_stack(x)

        if output_hidden_states:
            hidden_states = (token_embedding,) + hidden_states

        if not return_dict:
            return x, hidden_states

        return BaseModelOutput(
            last_hidden_state=x,
            hidden_states=hidden_states if output_hidden_states else None,
        )


class xLSTMForCausalLM(xLSTMPreTrainedModel, WeightDecayOptimGroupMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: xLSTMConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size

        self.model = xLSTMModel(config)

        self.lm_head = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
            bias=False,
        )

        self.post_init()
        # TODO: Add option for up-projection

    def get_input_embeddings(self):
        return self.model.token_embedding

    def set_input_embeddings(self, value: nn.Module):
        self.model.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def reset_parameters(self):
        self.model.xlstm_block_stack.reset_parameters()

        small_init_init_(
            self.get_input_embeddings().weight, dim=self.config.embedding_dim
        )

        if not self.config.tie_word_embeddings:
            small_init_init_(
                self.get_output_embeddings().weight, dim=self.config.embedding_dim
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output = self.model(
            input_ids,
            output_hidden_states=output_hidden_states,
        )

        hidden_state = output[0]

        logits = self.lm_head(hidden_state)
        logits = logits.float()

        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + output[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=output.hidden_states,
        )

    def step(
        self,
        idx: torch.Tensor,
        state: dict[str, dict[str, tuple[torch.Tensor, ...]]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, dict[str, tuple[torch.Tensor, ...]]]]:
        x = self.token_embedding(idx)
        x = self.emb_dropout(x)
        x, state = self.xlstm_block_stack.step(x, state=state, **kwargs)
        logits = self.lm_head(x)
        return logits, state

    def _create_weight_decay_optim_groups(
        self, **kwargs
    ) -> tuple[Sequence[nn.Parameter], Sequence[nn.Parameter]]:
        weight_decay, no_weight_decay = super()._create_weight_decay_optim_groups(
            **kwargs
        )
        # remove token embedding and add it to the correct group, accrording to the config
        weight_decay = list(weight_decay)
        removed = 0
        for idx in range(len(weight_decay)):
            if weight_decay[idx - removed] is self.get_input_embeddings().weight:
                weight_decay.pop(idx - removed)
                removed += 1
        weight_decay = tuple(weight_decay)

        # TODO: Fix this
        # if self.config.weight_decay_on_embedding:
        if True:
            weight_decay += (self.get_input_embeddings().weight,)
        else:
            no_weight_decay += (self.get_input_embeddings().weight,)

        return weight_decay, no_weight_decay

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = nn.Embedding(
            new_num_tokens, self.token_embedding.embedding_dim
        )
        self.token_embedding = new_embeddings.to(self.device)
        return new_embeddings

    def tie_weights(self):
        self.get_output_embeddings().weight = self.get_input_embeddings().weight

    def prepare_inputs_for_generation(
        self,
        input_ids,
        **kwargs,
    ):
        model_inputs = {
            "input_ids": input_ids.to(self.device),
        }
        return model_inputs


class xLSTMForSequenceClassification(xLSTMPreTrainedModel):

    def __init__(self, config: xLSTMConfig, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.model = xLSTMModel(config)
        self.classifier = nn.Linear(config.embedding_dim, config.num_labels, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output = self.model(
            input_ids,
            output_hidden_states=output_hidden_states,
        )

        hidden_state = output[0]

        logits = self.classifier(hidden_state)
        batch_size = input_ids.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1


        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None

        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        if not return_dict:
            output = (pooled_logits,) + output[1:]
            return ((loss,) + output) if loss is not None else output


        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            hidden_states=output.hidden_states,
        )
