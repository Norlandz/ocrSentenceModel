from pydantic import BaseModel

class HandwritingTextDto(BaseModel):
    imgDataUrl: str
    # lang: str = "EN"
    # text: str