from fastapi import APIRouter, Depends, Response, HTTPException
from fastapi.responses import JSONResponse

# from app.models import users as user_model
from src.model.HandwritingTextDto import HandwritingTextDto
from src.services.OcrService import OcrService


router = APIRouter(prefix="/ocr_text", tags=["text_ocr"])

@router.get("")
def default_route():
    return {"message": "Default Route"}

@router.post("/ocr_text", response_model=str)
async def ocr_text(oHandwritingTextDto: HandwritingTextDto, ocr_service: OcrService = Depends()):
    print(">> api ocr_text ", str(oHandwritingTextDto)[0:50])
    imgDataUrl = oHandwritingTextDto.imgDataUrl
    if imgDataUrl == "":
        raise HTTPException(status_code=400, detail="Input text is required")
    try:
        img = OcrService.convert_imgDataUrl2pilImg(imgDataUrl)
        # img.show()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Error convert_imgDataUrl2pilImg: {e}")
    try:
        textOcred = ocr_service.pred(img)
        print("textOcred:", textOcred)
        return textOcred
        # return Response(content="Example String", media_type="text/plain", status_code=200)
        # return JSONResponse(content=content, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing OCR: {e}")
