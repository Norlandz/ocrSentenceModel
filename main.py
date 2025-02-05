from fastapi import Depends, FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.services.OcrService import OcrService
from src.api.v1 import text_ocr


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup event
#     print("Application startup")
#     # Set configuration variables
#     app.state.my_config_value = "my_config"
#     yield
#     # Shutdown event
#     print("Application shutdown")
#
# app = FastAPI(lifespan=lifespan)

app = FastAPI()

origins = [
    # "http://localhost:3000",
    # not_working "http://127.0.0.1:5173",
    "http://localhost:5173",
    # "https://your-frontend-domain.com",
    # "*",  # If you want to allow all origins, use this, but avoid in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows the specific origins you set
    allow_credentials=True,  # Allows cookies to be sent
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, etc.)
    allow_headers=["*"],  # Allows all headers
)

ocr_service = OcrService()


def get_ocr_service():
    return ocr_service


app.include_router(text_ocr.router, dependencies=[Depends(get_ocr_service)])

# @app.get("/health")
# async def health_check():
#     return {"status": "ok"}

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run(app, host="127.0.0.1", port=8000)
