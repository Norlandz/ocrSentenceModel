from pathlib import Path

from PIL import Image
from PIL import ImageOps

from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from tqdm.notebook import tqdm

from src.mlConfig.Const import ctcBlankChar, idx2char, RGB_MEAN_IMAGENET, RGB_STD_IMAGENET, BW_MEAN_IMAGENET, BW_STD_IMAGENET

class IamTextLineDataset(Dataset):

    @staticmethod
    def parse_dict_imgPath2text(path_data_ImaTextLine_Mapping: Path, path_data_ImaTextLine_ImgRootFolder: Path):
        dict_imgPath2text: dict[Path, str] = {}

        with open(path_data_ImaTextLine_Mapping, "r") as f:
            for line in f:
                # Skip comments or blank lines
                if line.startswith("#") or line.strip() == "":
                    continue

                parts = line.strip().split()
                line_id = parts[0]  # e.g., a01-000u-00
                status = parts[1]  # e.g., ok
                text = " ".join(parts[8:]).replace("|", " ")  # Extract text, replace |

                list_LineFolderLv = line_id.split("-")

                if status == "ok":
                    image_path = path_data_ImaTextLine_ImgRootFolder / list_LineFolderLv[0] / f"{list_LineFolderLv[0]}-{list_LineFolderLv[1]}" / f"{line_id}.png"
                    dict_imgPath2text[image_path] = text  # Match img to text

        return dict_imgPath2text

    # def __init__(self, path_data_ImaTextLine_Mapping: Path, path_data_ImaTextLine_ImgRootFolder: Path):
    #     self.path_data_ImaTextLine_Mapping = path_data_ImaTextLine_Mapping
    #     self.path_data_ImaTextLine_ImgRootFolder = path_data_ImaTextLine_ImgRootFolder
    #     self.dict_imgPath2text = IamTextLineDataset.parse_dict_imgPath2text(path_data_ImaTextLine_Mapping, path_data_ImaTextLine_ImgRootFolder)
    #     self.list_imgPath2text = list(self.dict_imgPath2text.items())
    # //? order pb

    def __init__(self, list_imgPath2text: list[tuple[Path, str]], mode_ImgRgb: bool, imageInputSize: tuple[int, int]):
        self.list_imgPath2text = list_imgPath2text
        self.mode_ImgRgb = mode_ImgRgb
        self.imageInputSize = imageInputSize

    def __len__(self):
        return len(self.list_imgPath2text)

    def __getitem__(self, index: int):
        imgPath, text = self.list_imgPath2text[index]
        img = Image.open(imgPath)
        if self.mode_ImgRgb:
            img = img.convert("RGB")
        else:
            img = img.convert("L")
        imgTensor = self.transform(img)
        # filename = imgPath.stem
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
