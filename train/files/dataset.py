import logging
import random
import pandas as pd
import jsonlines
import json
import torch
import albumentations as A
import os
import numpy as np
import cv2

from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def base_masker(
    caption, vocab, mask_token="<MASK>", pad_token="<PAD>", ratio=0.15, *rest
):
    r"""Basic masking strategy, same as in BERT.
    Args:
        caption: str
        ratio: probability of token been masked out
    """
    (tokenizer,) = rest

    def measure_word_len(word):
        token_ids = tokenizer.encode(word)
        # tokens = [tokenizer.decode(x) for x in token_ids]
        return len(token_ids) - 2

    tokens = caption.split()
    bert_input_tokens = []
    output_mask = []
    bert_label_tokens = []  # 被 mask 的保留原词, 否则用 [PAD] 代替
    for i, token in enumerate(tokens):
        prob = random.random()
        if prob < ratio:
            prob /= ratio
            # 80% randomly change token to mask token
            if prob < 0.8:
                word_len = measure_word_len(token)
                bert_input_tokens += [mask_token] * word_len
            # 10% randomly change token to random token
            elif prob < 0.9:
                rand_token = random.choice(vocab).replace("</w>", "")
                word_len = measure_word_len(rand_token)
                # tokens[i] = random.randrange(self.tokenizer.vocab_size)
                bert_input_tokens += [rand_token]
            # 10% randomly change token to current token
            else:
                bert_input_tokens += [token]
                word_len = measure_word_len(token)
            output_mask += [1] * word_len
            bert_label_tokens += [token]
        else:
            word_len = measure_word_len(token)
            bert_input_tokens += [token]
            output_mask += [0] * word_len
            bert_label_tokens += [pad_token] * word_len

    logging.debug(f"\033[42moutput_mask:\033[0m {output_mask}")

    token_result = dict.fromkeys(
        ["bert_input_tokens", "output_mask", "bert_label_tokens"], None
    )
    for key in token_result:
        token_result[key] = eval(key)  # HACK dark magic, could be dangerous
    return token_result


def encode_mlm(
    caption, vocab, mask_token: str, pad_token: str, ratio: float, tokenizer, cfg
):
    r"""Process captions into masked input and ground truth
    Args:
        caption:
        vocab:
        mask_token:
        pad_token:
        ratio:
        tokenizer:
        args:
    Return:
        bert_input:
        bert_label:

    Reference Code:
    - [BERT-pytorch]()
    - [DataCollatorForWholeWordMask](https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L1072)
    """
    context_length = cfg.text_model.context_length
    masker = base_masker

    token_result = masker(
        caption,
        vocab,
        mask_token,
        pad_token,
        ratio,
        tokenizer,
    )  # Remove words for MLM task

    output_mask = token_result["output_mask"]
    output_mask += [0] * (context_length - len(output_mask))
    output_mask = torch.tensor(output_mask[:context_length])
    logging.debug(len(output_mask), output_mask)

    bert_input_tokens = token_result["bert_input_tokens"]
    bert_input = " ".join(bert_input_tokens)
    bert_label_tokens = token_result["bert_label_tokens"]
    bert_label = " ".join(bert_label_tokens)

    return bert_input, bert_label


class ImgTextDataset(Dataset):
    def __init__(self, cfg, transforms, phase):
        logging.debug(f"Loading json file")

        self.cfg = cfg
        if phase == "train":
            json_path = cfg.data.train_json_path
        if phase == "valid":
            json_path = cfg.data.valid_json_path

        if json_path[-4:] == "json":
            with open(json_path) as json_file:
                self.json = json.load(json_file)
        elif json_path[-5:] == "jsonl":
            with open(json_path, "r") as json_file:
                self.json = [eval(x) for x in list(json_file)]
        else:
            assert False, "Incorrect Json File Extension"

        self.transforms = transforms

        if cfg.text_model.mlm:
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg.text_model.bert_model_name
            )
            self.mask_token, self.pad_token = "[MASK]", "[PAD]"
            vocab = list(self.tokenizer.get_vocab().keys())
            # Remove special token from vocab
            self.vocab_with_no_special_token = [
                vocab_token
                for vocab_token in vocab
                if vocab_token not in self.tokenizer.all_special_tokens
            ]
            self.ratio = cfg.text_model.mask_ratio if phase == "train" else 0.0

        logging.debug("Done loading data.")

    def __len__(self):
        return len(self.json)

    def __getitem__(self, idx):
        output = dict.fromkeys(["images", "bert_input", "bert_label"], None)
        image_path = os.path.join(
            self.cfg.data.image_main_path, self.json[idx]["image"]
        )
        img = Image.open(image_path)
        img = np.array(img)
        img = self.transforms(image=img)["image"]
        if len(img.shape) == 3:
            img = np.transpose(img, (2, 0, 1))
        else:
            img = img[np.newaxis, ...]
            img = np.concatenate([img, img, img], axis=0)
        img = np.asarray(img).astype(np.float32)

        img = img - img.min()
        if img.max() == 0:
            print(f"problematic image path : {image_path}")
        img = img / img.max()

        img = torch.tensor(img)

        caption = str(self.json[idx]["caption"])
        bert_input, bert_label = encode_mlm(
            caption=caption,
            vocab=self.vocab_with_no_special_token,
            mask_token=self.mask_token,
            pad_token=self.pad_token,
            ratio=self.ratio,
            tokenizer=self.tokenizer,
            cfg=self.cfg,
        )

        output.update(
            {"images": img, "bert_input": bert_input, "bert_label": bert_label}
        )
        return output


class Transforms:
    def __init__(self, cfg):
        self.train_transforms = A.Compose(
            [
                A.augmentations.geometric.resize.Resize(
                    height=cfg.data.image_size[0], width=cfg.data.image_size[1]
                )
            ]
        )
        self.valid_transforms = A.Compose(
            [
                A.augmentations.geometric.resize.Resize(
                    height=cfg.data.image_size[0], width=cfg.data.image_size[1]
                )
            ]
        )

        self.train_transforms_heavy = A.Compose(
            [
                # flip
                A.augmentations.geometric.resize.Resize(
                    height=cfg.data.image_size[0], width=cfg.data.image_size[1]
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # # downscale
                A.OneOf(
                    [
                        A.Downscale(
                            scale_min=0.75,
                            scale_max=0.95,
                            interpolation=dict(
                                upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_AREA
                            ),
                            p=0.1,
                        ),
                        A.Downscale(
                            scale_min=0.75,
                            scale_max=0.95,
                            interpolation=dict(
                                upscale=cv2.INTER_LANCZOS4, downscale=cv2.INTER_AREA
                            ),
                            p=0.1,
                        ),
                        A.Downscale(
                            scale_min=0.75,
                            scale_max=0.95,
                            interpolation=dict(
                                upscale=cv2.INTER_LINEAR, downscale=cv2.INTER_LINEAR
                            ),
                            p=0.8,
                        ),
                    ],
                    p=0.125,
                ),
                A.OneOf(
                    [
                        A.GridDistortion(
                            num_steps=5,
                            distort_limit=0.3,
                            interpolation=cv2.INTER_LINEAR,
                            border_mode=cv2.BORDER_CONSTANT,
                            value=0,
                            mask_value=None,
                            normalized=True,
                            p=0.2,
                        ),
                    ],
                    p=0.5,
                ),
            ],
            p=1,
        )
