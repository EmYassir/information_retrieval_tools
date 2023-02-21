from functools import partial
import sys
import torch
import logging
from torch import nn as nn
from torch.autograd import Variable
from dataclasses import dataclass


from typing import (
    Union,
    Tuple,
    Optional,
)

from torch import sigmoid, softmax
from torch.nn import Identity

from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import ModelOutput

from src.config.gan_config import GANConfig, FullGANConfig


## DEBUG ONLY
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class GeneralGANOutput(ModelOutput):
    """
    Class for outputs of [`GenModel`, `DisModel`].
    Args:
        output_distribution (`torch.FloatTensor` of shape `(batch_size, n_documents)`):
            The encoder outputs the *output_distribution* that corresponds to the documents that likely have an answer.
        last_hidden_state (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, n_docs, sequence_length, hidden_size)`.  Hidden-states of the model at the output of each layer plus the initial embedding outputs.
    """

    output_distribution: torch.FloatTensor
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class FullGANOutput(ModelOutput):
    """
    Class for outputs of the full GAN model.
    Args:
        generator_output (ModelOutput):
            The output of the generator.
        discriminator_output (ModelOutput):
            The output of the generator.
        ans_discriminator_output (ModelOutput):
            The output of the generator.
    """

    generator_output: Optional[ModelOutput] = None
    discriminator_output: Optional[ModelOutput] = None
    ans_discriminator_output: Optional[ModelOutput] = None


"""""" """""" """""" """""" """""" """""
    Main GAN models' interfaces 
""" """""" """""" """""" """""" """""" ""


class GANModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization.
    """

    config_class = GANConfig
    load_tf_weights = None
    base_model_prefix = "bert_based_gan_model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Override main method to avoid some issues
    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        self.tie_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

    def move_to(self, device):
        return self.to(device)

    @classmethod
    def load_from_disk(cls, dir_path):
        return cls.from_pretrained(dir_path)

    def save_to_disk(self, dir_path):
        # Save the model
        self.save_pretrained(dir_path)


class DefaultModel(GANModel):
    base_model_prefix = "bert_based_generator_model"

    def __init__(self, config: GANConfig):
        super().__init__(config)

        # Intialize simple fields
        self.model_path = config.model_path
        self.model_name = config.model_name
        if config.activation == "softmax":
            self.activation = partial(softmax, dim=-1)
        elif config.activation == "sigmoid":
            self.activation = sigmoid
        else:
            self.activation = Identity()
        # Loading the model
        if self.model_path is not None and len(self.model_path) > 0:
            self.bert_model = BertModel.from_pretrained(self.model_path)
        elif self.model_name is not None and len(self.model_name) > 0:
            self.bert_model = BertModel.from_pretrained(
                self.model_name,
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            self.bert_model = BertModel(
                BertConfig.from_dict(config.model_cfg), add_pooling_layer=False
            )
        if self.bert_model.config.hidden_size <= 0:
            raise ValueError(
                f"Encoder hidden_size ({self.bert_model.config.hidden_size}) should be positive"
            )

        # The projection layer
        self.encode_proj = nn.Linear(self.bert_model.config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.IntTensor,
        attention_mask: Optional[torch.IntTensor] = None,
        token_type_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        n_documents: int = 100,
        k_select: int = 0,
        **kwargs,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.FloatTensor, ...]]:
        n_documents = input_ids.size(1)
        batch_size = input_ids.size(0)

        # Reshape the batch (batch, n_docs, seq_len) -> (batch * n_docs, seq_len)
        input_ids = input_ids.reshape((batch_size * n_documents, -1))
        attention_mask = (
            attention_mask.reshape((batch_size * n_documents, -1))
            if attention_mask is not None
            else None
        )
        token_type_ids = (
            token_type_ids.reshape((batch_size * n_documents, -1))
            if token_type_ids is not None
            else None
        )
        inputs_embeds = (
            inputs_embeds.reshape((batch_size * n_documents, -1))
            if inputs_embeds is not None
            else None
        )
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Input ids dimension: (batch * n_docs, seq_len, hid_size)

        if n_documents <= 0:
            raise ValueError(
                f"The number of documents should be positive (found '{n_documents}'\)."
            )
        if n_documents < k_select:
            raise ValueError(
                f"The value of 'k_select' should be lesser than the value of '{n_documents}'\."
            )
        seq_len, hidden_size = outputs[0].size(1), outputs[0].size(2)
        hidden_states = outputs[0].reshape(
            batch_size, n_documents, seq_len, hidden_size
        )  # hidden states (batch, n_docs, seq_len, hid_size)

        cls_output = hidden_states[:, :, 0, :].reshape(
            batch_size, n_documents, -1
        )  # CLS outputs (batch, n_docs, hid_size)

        # Projection
        cls_output = self.encode_proj(cls_output).squeeze(
            -1
        )  # CLS outputs (batch, n_docs)
        cls_output = self.activation(cls_output)  # CLS outputs (batch, n_docs)

        if k_select > 0:
            # The pooled output will be built as follows:
            #   1) output[...,...] != 0. if it corresponds to a  topk element
            #   2) output[...,...] == 0. otherwise
            topk, indices = torch.topk(cls_output, k_select)
            # aux_output = torch.autograd.Variable(torch.zeros_like(pooled_output, dtype=pooled_output.dtype, device=pooled_output.dtype))
            aux_output = torch.zeros_like(
                cls_output, dtype=cls_output.dtype, device=cls_output.device
            )
            cls_output = aux_output.scatter(1, indices, topk)

        if not return_dict:
            return (hidden_states, cls_output)

        return GeneralGANOutput(
            last_hidden_state=hidden_states,
            output_distribution=cls_output,
        )

    @property
    def embeddings_size(self) -> int:
        return self.bert_model.config.hidden_size

    def move_to(self, device):
        return self.to(device)


"""""" """""" """""" """""" """
    FULL GAN implementation 
""" """""" """""" """""" """"""


class FullGANInterface(PreTrainedModel):
    """
    An abstract class to handle weights initialization.
    """

    config_class = FullGANConfig
    load_tf_weights = None
    base_model_prefix = "full_gan_interface"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        # We have some nested pretrained modules...
        if isinstance(module, PreTrainedModel):
            pass  # These modules have been initialized already
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Override main method to avoid some issues
    def init_weights(self):
        # Initialize weights
        self.apply(self._init_weights)

        # Tie weights should be skipped when not initializing all weights
        # since from_pretrained(...) calls tie weights anyways
        self.tie_weights()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value

    @classmethod
    def load_from_disk(cls, dir_path):
        return cls.from_pretrained(dir_path)

    def save_to_disk(self, dir_path):
        # Save the model
        self.save_pretrained(dir_path)


""" Full gan framework """


class FullGANModel(FullGANInterface):
    base_model_prefix = "full_gan_model"

    def __init__(self, config: FullGANConfig):
        super().__init__(config)

        # Generator
        subconfig = GANConfig.from_dict(config.generator_cfg)
        self.generator = DefaultModel(subconfig)
        subconfig = GANConfig.from_dict(config.discriminator_cfg)
        self.discriminator = DefaultModel(subconfig)
        subconfig = GANConfig.from_dict(config.ans_discriminator_cfg)
        self.ans_discriminator = DefaultModel(subconfig)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.IntTensor,
        attention_mask: Optional[torch.IntTensor] = None,
        token_type_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        k_select: int = 0,
        options: str = "gad",
        **kwargs,
    ) -> Union[BaseModelOutputWithPooling, Tuple[torch.FloatTensor, ...]]:
        ret = {}
        # 'g' for Generator
        if "g" in options or "G" in options:
            ret["generator_output"] = self.generator(
                input_ids,
                attention_mask,
                token_type_ids,
                inputs_embeds,
                output_attentions,
                output_hidden_states,
                return_dict,
                k_select,
            )
        # 'd' for Discriminator
        if "d" in options or "D" in options:
            ret["discriminator_output"] = self.discriminator(
                input_ids,
                attention_mask,
                token_type_ids,
                inputs_embeds,
                output_attentions,
                output_hidden_states,
                return_dict,
                k_select,
            )

        # 'a' for Answwer-discriminator
        if "a" in options or "A" in options:
            ret["ans_discriminator_output"] = self.ans_discriminator(
                input_ids,
                attention_mask,
                token_type_ids,
                inputs_embeds,
                output_attentions,
                output_hidden_states,
                return_dict,
                k_select,
            )

        if not return_dict:
            tuple_ret = ()
            for key in [
                "generator_output",
                "discriminator_output",
                "ans_discriminator_output",
            ]:
                if key in ret:
                    tuple_ret = (tuple_ret, ret[key])
            return tuple_ret

        return FullGANOutput(
            generator_output=ret["generator_output"]
            if "generator_output" in ret
            else None,
            discriminator_output=ret["discriminator_output"]
            if "discriminator_output" in ret
            else None,
            ans_discriminator_output=ret["ans_discriminator_output"]
            if "ans_discriminator_output" in ret
            else None,
        )

    @property
    def embeddings_size(self) -> int:
        return self.generator.bert_model.config.hidden_size

    def move_to(self, device):
        return self.to(device)
