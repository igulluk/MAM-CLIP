import torch
import torch.nn as nn
import timm
import numpy as np
import logging
import random
import pandas as pd
import hydra
import lightning.pytorch as pl
import os
import torch.nn.functional as F
import json

from PIL import Image
from transformers import AutoTokenizer, AutoModel
from files.nnblocks import LayerNorm, Transformer
from dataclasses import dataclass


class VLModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = timm.create_model(
            cfg.vision_model.name,
            pretrained=True,
        )
        in_features = self.vision_encoder.head.fc.in_features
        self.vision_encoder.head.fc = nn.Linear(in_features, cfg.embed_dim)

        assert (
            cfg.text_model.bert_model_name
            == "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        ), "Please check [CLS]'s token id"
        self.cls_id = 2  # [CLS]'s token id is 2, while it varies from tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_model.bert_model_name)
        self.text_encoder = AutoModel.from_pretrained(cfg.text_model.bert_model_name)

        self.transformer_width = cfg.text_model.width
        self.positional_embedding = nn.Parameter(
            torch.empty(cfg.text_model.context_length, cfg.text_model.width)
        )
        self.ln_final = LayerNorm(cfg.text_model.width)

        self.context_length = cfg.text_model.context_length

        self.text_projection = nn.Parameter(
            torch.empty(cfg.text_model.width, cfg.embed_dim)
        )
        self.mlm_projection = None
        if cfg.text_model.mlm:
            self.mlm_projection = nn.Parameter(
                torch.empty(cfg.text_model.width, cfg.text_model.vocab_size)
            )
        self.softmax = nn.LogSoftmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.img_special_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.fusion_module = Transformer(
            width=cfg.text_model.width,
            layers=cfg.text_model.fusion_layers,
            heads=cfg.text_model.heads,
        )
        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def init_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if hasattr(self.vision_encoder, "init_parameters"):
            self.vision_encoder.init_parameters()

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer_width**-0.5)
        if self.mlm_projection is not None:
            nn.init.normal_(self.mlm_projection, std=self.transformer_width**-0.5)

    def encode_image(self, image):
        return self.vision_encoder(image)

    def encode_text(self, batch, image_features):
        encoded_input = self.tokenizer(
            batch["bert_input"],
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
            return_tensors="pt",
        )
        encoded_label = self.tokenizer(
            batch["bert_label"],
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
            return_tensors="pt",
        )
        encoded_label = encoded_label["input_ids"].to(image_features.device)
        encoded_input["input_ids"] = encoded_input["input_ids"].to(
            image_features.device
        )  # [128, 77]

        x = self.text_encoder(
            input_ids=encoded_input["input_ids"], output_attentions=False
        )
        x = x["last_hidden_state"]

        inside = (encoded_input["input_ids"] == self.cls_id).squeeze()
        if len(inside.shape) == 1:
            inside = inside.unsqueeze(0)
        last_token_index = torch.nonzero(inside)
        text_features = x[torch.arange(x.shape[0]), last_token_index[:, 1]]

        text_features = text_features @ self.text_projection  # NOTE for matching

        # Fusion Module
        img = torch.unsqueeze(image_features, 1)  # [128, 1 ,768]
        B, _len, _dim = x.shape
        img_special_tokens = self.img_special_token.expand(
            B, -1, -1
        )  # [128, 1, embed_dim]
        x = torch.cat([x, img_special_tokens, img], dim=1)  # [128, 77+1+1, 768]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.fusion_module(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, :-2, :]  # Remove token [img_special_token, img]

        bert_prediction = None
        if self.cfg.text_model.mlm:
            bert_prediction = self.softmax(
                x @ self.mlm_projection
            )  # [batch_size=128, n_ctx=77, vocab_size=49409]

        attentions = None
        text_output = dict.fromkeys(
            ["text_features", "bert_prediction", "attentions", "encoded_label"], None
        )
        for key in text_output:
            text_output[key] = eval(key)  # HACK dark magic, could be dangerous

        return text_output

    def forward(self, batch):
        image = batch["images"]
        # image = image.to(device=self.device, non_blocking=True)

        if (image is None) or (batch["bert_input"] is None):
            raise RuntimeError("Missing Image OR Text in the input")

        image_features = self.encode_image(image)

        image_features = F.normalize(image_features, dim=-1)  # [128, 768]

        text_output = self.encode_text(batch, image_features)
        text_features = F.normalize(text_output["text_features"], dim=-1)

        clip_prediction = dict.fromkeys(
            [
                "image_features",
                "text_features",
                "logit_scale",
                "bert_label",
                "bert_prediction",
                "attentions",
            ],
            None,
        )
        clip_prediction.update(
            {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
                "bert_label": text_output["encoded_label"],
                "bert_prediction": text_output["bert_prediction"],
                "attentions": text_output["attentions"],
            }
        )
        return clip_prediction


class lightningModel(pl.LightningModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.VLModel = VLModel(cfg)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.mlm_coeff = 0.5

    def training_step(self, batch, batch_idx):
        output = self.VLModel.forward(batch)
        match_loss, mlm_loss = self.loss(output)
        loss = match_loss + self.mlm_coeff * mlm_loss
        step_out = {
            "loss": loss.detach(),
            "match_loss": match_loss.detach(),
            "mlm_loss": mlm_loss.detach(),
        }
        self.training_step_outputs.append(step_out)
        return loss

    def on_train_epoch_end(self):

        loss = torch.stack([x["loss"] for x in self.training_step_outputs]).mean()
        match_loss = torch.stack(
            [x["match_loss"] for x in self.training_step_outputs]
        ).mean()
        mlm_loss = torch.stack(
            [x["mlm_loss"] for x in self.training_step_outputs]
        ).mean()
        logs = {
            "train_loss": loss,
            "train_match_loss": match_loss,
            "train_mlm_loss": mlm_loss,
        }
        self.log_dict(logs, sync_dist=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output = self.VLModel.forward(batch)
        match_loss, mlm_loss = self.loss(output)
        loss = match_loss + self.mlm_coeff * mlm_loss
        step_out = {
            "loss": loss.detach(),
            "match_loss": match_loss.detach(),
            "mlm_loss": mlm_loss.detach(),
        }
        self.validation_step_outputs.append(step_out)
        return loss

    def on_validation_epoch_end(self):

        loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        match_loss = torch.stack(
            [x["match_loss"] for x in self.validation_step_outputs]
        ).mean()
        mlm_loss = torch.stack(
            [x["mlm_loss"] for x in self.validation_step_outputs]
        ).mean()
        logs = {
            "valid_loss": loss,
            "valid_match_loss": match_loss,
            "valid_mlm_loss": mlm_loss,
        }
        self.log_dict(logs, sync_dist=True)
        self.validation_step_outputs.clear()

    def loss(self, output):

        image_features = output["image_features"]
        text_features = output["text_features"]
        logit_scale = output["logit_scale"]
        bert_prediction = output["bert_prediction"]

        bert_label = output["bert_label"]

        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(
            num_logits, dtype=torch.long, device=logits_per_image.device
        )

        mlm_loss = F.nll_loss(
            bert_prediction.transpose(1, 2), bert_label, ignore_index=0
        )
        match_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return match_loss, mlm_loss

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.cfg.optimizer, params=self.parameters()
        )
        return optimizer
