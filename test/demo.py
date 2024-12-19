import os
import sys
from pathlib import Path
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# print(os.path.dirname(os.path.realpath(__file__)))
# sys.path.append("C:/usp/usehpsj/proj/ocrSentenceModel")
sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from PIL import Image
from src.config.path import get_project_root
from src.services.OcrService import OcrService

# C:/usp/usehpsj/study/textbook/DlPytorch/code/RnnLearn/04_crnnIamHwTextLine/data/original/fki.tic.heia-fr.ch/iam-handwriting-database/lines/a01/a01-000u/a01-000u-00.png

oOcrService = OcrService()

imgPath = get_project_root() / "data/original/fki.tic.heia-fr.ch/iam-handwriting-database/lines/a01/a01-000u/a01-000u-00.png"
img = Image.open(imgPath)

textOcred = oOcrService.pred(img)
print(textOcred)