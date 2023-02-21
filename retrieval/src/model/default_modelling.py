import os
import torch
from torch import nn as nn
from torch.nn import functional as F


from typing import Union, Tuple, Dict, List, Optional

from transformers import BertConfig, BertForSequenceClassification

from transformers.modeling_outputs import SequenceClassifierOutput

from src.config.gan_config import GANConfig


## DEBUG ONLY
import sys
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

"""""" """""" """""" """""" """""" """""
    Main GAN models' interfaces 
""" """""" """""" """""" """""" """""" ""
""" Main interface """


class DefaultModel(nn.Module):
    def __init__(self, **kwargs):
        super(DefaultModel, self).__init__()

    @classmethod
    def from_config(cls, config: GANConfig):
        return cls(**config.to_dict())

    def forward(self, **kwargs):
        pass

    def move_to(self, device):
        return self.to(device)

    def load_weights_from_disk(self, dir_path):
        pass

    def save_to_disk(self, dir_path):
        pass


"""""" """""" """""" """""" """
    GEN implementation 
""" """""" """""" """""" """"""
""" Generator """


class GenModel(DefaultModel):
    def __init__(
        self,
        model_cfg: Optional[Dict],
        model_path: Optional[str],
        model_name: Optional[str],
        **kwargs,
    ):
        super(GenModel, self).__init__()

        # Creating the config file
        self.config = GANConfig(model_cfg, model_path, model_name)

        self.n_classes = self.config.n_classes

        # Loading the model
        if model_path is not None:
            self.scorer = BertForSequenceClassification.from_pretrained(model_path)
        elif self.config.model_name is not None:
            self.scorer = BertForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.n_classes,
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            config = BertConfig.from_dict(model_cfg)
            config.output_attentions = False
            config.output_hidden_states = False
            self.scorer = BertForSequenceClassification(config)

    @classmethod
    def from_config(cls, config: GANConfig):
        return cls(**config.to_dict())

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # return self.scorer(
        #     input_ids,
        #     attention_mask,
        #     token_type_ids,
        #     position_ids,
        #     head_mask,
        #     inputs_embeds,
        #     labels,
        #     output_attentions,
        #     output_hidden_states,
        #     return_dict,
        # )
        return torch.softmax(torch.rand(kwargs["answerability_labels"].shape), dim=1)

    def move_to(self, device):
        return self.to(device)

    def load_weights_from_disk(self, dir_path):
        self.scorer = BertForSequenceClassification.from_pretrained(dir_path)

    def save_to_disk(self, dir_path):
        # Making output directory
        os.makedirs(dir_path, exist_ok=True)
        # A sub directory where BERT model is saved
        self.scorer.save_pretrained(dir_path)
        # Saving the config file
        self.config.to_json_file(os.path.join(dir_path, "global_config.json"))


"""""" """""" """""" """""" """
    RDIS implementation 
""" """""" """""" """""" """"""
""" Rank Discrminator """


class RankDISModel(DefaultModel):
    def __init__(
        self,
        model_cfg: Optional[Dict],
        model_path: Optional[str],
        model_name: Optional[str],
        **kwargs,
    ):
        super(RankDISModel, self).__init__()

        # Creating the config file
        self.config = GANConfig(model_cfg, model_path, model_name)

        # Loading the model
        if model_path is not None:
            self.scorer = BertForSequenceClassification.from_pretrained(model_path)
        elif self.config.model_name is not None:
            self.scorer = BertForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.n_classes,
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            config = BertConfig.from_dict(model_cfg)
            config.output_attentions = False
            config.output_hidden_states = False
            self.scorer = BertForSequenceClassification(config)

    @classmethod
    def from_config(cls, config: GANConfig):
        return cls(**config.to_dict())

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # return self.scorer(
        #     input_ids,
        #     attention_mask,
        #     token_type_ids,
        #     position_ids,
        #     head_mask,
        #     inputs_embeds,
        #     labels,
        #     output_attentions,
        #     output_hidden_states,
        #     return_dict,
        # )
        return torch.softmax(torch.rand(kwargs["answerability_labels"].shape), dim=1)

    def move_to(self, device):
        return self.to(device)

    def load_weights_from_disk(self, dir_path):
        self.scorer = BertForSequenceClassification.from_pretrained(dir_path)

    def save_to_disk(self, dir_path):
        # Making output directory
        os.makedirs(dir_path, exist_ok=True)
        # A sub directory where BERT model is saved
        self.scorer.save_pretrained(dir_path)
        # Saving the config file
        self.config.to_json_file(os.path.join(dir_path, "global_config.json"))


"""""" """""" """""" """""" """
    RDIS implementation 
""" """""" """""" """""" """"""
""" Rank Discrminator """


class RankDISModel(DefaultModel):
    def __init__(
        self,
        model_cfg: Optional[Dict],
        model_path: Optional[str],
        model_name: Optional[str],
        **kwargs,
    ):
        super(RankDISModel, self).__init__()

        # Creating the config file
        self.config = GANConfig(model_cfg, model_path, model_name)

        # Loading the model
        if model_path is not None:
            self.scorer = BertForSequenceClassification.from_pretrained(model_path)
        elif self.config.model_name is not None:
            self.scorer = BertForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.n_classes,
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            config = BertConfig.from_dict(model_cfg)
            config.output_attentions = False
            config.output_hidden_states = False
            self.scorer = BertForSequenceClassification(config)

    @classmethod
    def from_config(cls, config: GANConfig):
        return cls(**config.to_dict())

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # return self.scorer(
        #     input_ids,
        #     attention_mask,
        #     token_type_ids,
        #     position_ids,
        #     head_mask,
        #     inputs_embeds,
        #     labels,
        #     output_attentions,
        #     output_hidden_states,
        #     return_dict,
        # )
        return torch.rand(kwargs["answerability_labels"].shape)

    def move_to(self, device):
        return self.to(device)

    def load_weights_from_disk(self, dir_path):
        self.scorer = BertForSequenceClassification.from_pretrained(dir_path)

    def save_to_disk(self, dir_path):
        # Making output directory
        os.makedirs(dir_path, exist_ok=True)
        # A sub directory where BERT model is saved
        self.scorer.save_pretrained(dir_path)
        # Saving the config file
        self.config.to_json_file(os.path.join(dir_path, "global_config.json"))


"""""" """""" """""" """""" """
    ADIS implementation 
""" """""" """""" """""" """"""
""" Answerability Discrminator """


class AnsDISModel(DefaultModel):
    def __init__(
        self,
        model_cfg: Optional[Dict],
        model_path: Optional[str],
        model_name: Optional[str],
        **kwargs,
    ):
        super(AnsDISModel, self).__init__()

        # Creating the config file
        self.config = GANConfig(model_cfg, model_path, model_name)

        # Loading the model
        if model_path is not None:
            self.scorer = BertForSequenceClassification.from_pretrained(model_path)
        elif self.config.model_name is not None:
            self.scorer = BertForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.n_classes,
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            config = BertConfig.from_dict(model_cfg)
            config.output_attentions = False
            config.output_hidden_states = False
            self.scorer = BertForSequenceClassification(config)

    @classmethod
    def from_config(cls, config: GANConfig):
        return cls(**config.to_dict())

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # return self.scorer(
        #     input_ids,
        #     attention_mask,
        #     token_type_ids,
        #     position_ids,
        #     head_mask,
        #     inputs_embeds,
        #     labels,
        #     output_attentions,
        #     output_hidden_states,
        #     return_dict,
        # )
        return torch.rand(kwargs["answerability_labels"].shape)

    def move_to(self, device):
        return self.to(device)

    def load_weights_from_disk(self, dir_path):
        self.scorer = BertForSequenceClassification.from_pretrained(dir_path)

    def save_to_disk(self, dir_path):
        # Making output directory
        os.makedirs(dir_path, exist_ok=True)
        # A sub directory where BERT model is saved
        self.scorer.save_pretrained(dir_path)
        # Saving the config file
        self.config.to_json_file(os.path.join(dir_path, "global_config.json"))
