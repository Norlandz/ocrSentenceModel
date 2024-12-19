import re

import time
from datetime import datetime

import random

from pathlib import Path
import os
import sys
import glob
from io import StringIO

from enum import Enum
from dataclasses import dataclass
from typing import TypedDict, Callable

import warnings
import traceback

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from PIL import Image
from PIL import ImageOps

import torch
from torch import Tensor
from torch import optim
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet18

from src.mlConfig.Const import ctcBlankChar, idx2char, RGB_MEAN_IMAGENET, RGB_STD_IMAGENET, BW_MEAN_IMAGENET, BW_STD_IMAGENET

class CAPTCHADataset(Dataset):
    def __init__(self, list_imgPath: list[Path], mode_ImgRgb: bool, imageInputSize: tuple[int, int]):
        self.list_imgPath = list_imgPath
        self.mode_ImgRgb = mode_ImgRgb
        self.imageInputSize = imageInputSize

    def __len__(self):
        return len(self.list_imgPath)

    def __getitem__(self, index: int):
        imgPath = self.list_imgPath[index]
        img = Image.open(imgPath)
        if self.mode_ImgRgb:
            img = img.convert("RGB") # ~~~~// dont put inside transform.. make slow
        else:
            img = img.convert("L")
        imgTensor = self.transform(img)
        text = imgPath.stem
        return imgTensor, text

    def transform(self, img: Image.Image) -> Tensor:
        if self.mode_ImgRgb:
            img = ImageOps.pad(img, self.imageInputSize, color=(255, 255, 255))
            normalize_mean = RGB_MEAN_IMAGENET
            normalize_std = RGB_STD_IMAGENET
        else:
            img = ImageOps.pad(img, self.imageInputSize, color=255)
            normalize_mean = BW_MEAN_IMAGENET
            normalize_std = BW_STD_IMAGENET
        transform_ops = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std),
            ]
        )
        imgTensor: Tensor = transform_ops(img) # type: ignore
        return imgTensor