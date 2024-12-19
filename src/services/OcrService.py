from io import BytesIO
import re

import time
from datetime import datetime

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from PIL import Image, ImageOps
from torchvision import transforms

from src.config.path import get_project_root
from src.dataset.IamTextLineDataset import IamTextLineDataset

from src.mlModel.Crnn import CRNN
from src.mlConfig.Const import (
    ctcBlankChar,
    idx2char,
    RGB_MEAN_IMAGENET,
    RGB_STD_IMAGENET,
    BW_MEAN_IMAGENET,
    BW_STD_IMAGENET,
    mode_ImgRgb,
    mode_UsePretrainedWeight,
    imageInputSize,
    num_chars,
    rnn_hidden_size,
)

from base64 import b64decode


class DatasetTransform:
    @staticmethod
    def transform(img: Image.Image) -> Tensor:
        if mode_ImgRgb:
            img = img.convert("RGB")
            img = ImageOps.pad(img, imageInputSize, color=(255, 255, 255))
            normalize_mean = RGB_MEAN_IMAGENET
            normalize_std = RGB_STD_IMAGENET
        else:
            img = img.convert("L")
            img = ImageOps.pad(img, imageInputSize, color=255)
            normalize_mean = BW_MEAN_IMAGENET
            normalize_std = BW_STD_IMAGENET
        transform_ops = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std),
            ]
        )
        imgTensor: Tensor = transform_ops(img)  # type: ignore
        return imgTensor


class OcrService:

    @staticmethod
    def convert_textRnn2textRepeat(predRnnSeqOutDistLogit_text_batch: Tensor) -> list[str]:
        text_batch_tokens = F.softmax(predRnnSeqOutDistLogit_text_batch, 2).argmax(2)  # [T, batch_size]
        text_batch_tokens = text_batch_tokens.numpy().T  # [batch_size, T]

        text_batch_tokens_new = []
        for text_tokens in text_batch_tokens:
            text = [idx2char[idx] for idx in text_tokens]
            text = "".join(text)
            text_batch_tokens_new.append(text)

        return text_batch_tokens_new

    @staticmethod
    def remove_duplicates(text):
        if len(text) > 1:
            letters = [text[0]] + [letter for idx, letter in enumerate(text[1:], start=1) if text[idx] != text[idx - 1]]
        elif len(text) == 1:
            letters = [text[0]]
        else:
            return ""
        return "".join(letters)

    @staticmethod
    def convert_textRepeat2text(word: str):
        parts = word.split(ctcBlankChar)
        parts = [OcrService.remove_duplicates(part) for part in parts]
        corrected_word = "".join(parts)
        return corrected_word

    # @staticmethod
    # def pred(dataset: Dataset):
    #     df_TargetVsPred = pd.DataFrame(columns=["target", "pred", "predRepeat"])
    #     dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    #     list_target_text = []
    #     list_pred_text_batch_Repeat = []
    #     list_pred_text_batch = []
    #     with torch.no_grad():
    #         for imgTensor_batch, target_text_batch in tqdm(dataloader, leave=True):
    #             predRnnSeqOutDistLogit_text_batch = crnn(imgTensor_batch.to(device))  # [T, batch_size, num_classes==num_features]
    #             pred_text_batch_Repeat = convert_textRnn2textRepeat(predRnnSeqOutDistLogit_text_batch.cpu())
    #             list_target_text = list_target_text + list(target_text_batch)
    #             list_pred_text_batch_Repeat += pred_text_batch_Repeat
    #             list_pred_text_batch += [convert_textRepeat2text(text) for text in pred_text_batch_Repeat]
    #     df_TargetVsPred["target"] = list_target_text
    #     df_TargetVsPred["pred"] = list_pred_text_batch
    #     df_TargetVsPred["predRepeat"] = list_pred_text_batch_Repeat
    #     return df_TargetVsPred

    def __init__(self) -> None:
        self.crnn = CRNN(num_chars, mode_ImgRgb, rnn_hidden_size=rnn_hidden_size)
        weights = torch.load(Path(get_project_root() / "model/crnn - iamTextLine - 2024_1212_2215_24 epoch6.pth"))["model_state_dict"]
        self.crnn.load_state_dict(weights)
        self.crnn.eval()

    @staticmethod
    def convert_imgDataUrl2pilImg(imgDataUrl: str):
        # Remove data URI prefix
        image_data = imgDataUrl.split(",")[1]

        # Decode base64
        image_bytes = b64decode(image_data)

        img = Image.open(BytesIO(image_bytes))
        return img

    def pred(self, img: Image.Image):
        pred_text: str
        with torch.no_grad():
            imgTensor = DatasetTransform.transform(img)
            imgTensor_batch = imgTensor.unsqueeze(0)
            # print(imgTensor_batch.shape)
            predRnnSeqOutDistLogit_text_batch = self.crnn(imgTensor_batch)  # [T, batch_size, num_classes==num_features]
            pred_text_batch_Repeat = OcrService.convert_textRnn2textRepeat(predRnnSeqOutDistLogit_text_batch)
            pred_text = OcrService.convert_textRepeat2text(pred_text_batch_Repeat[0])
        return pred_text
