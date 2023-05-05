
from fastapi import FastAPI, File, UploadFile
import shutil
import cv2
from fastapi.middleware.cors import CORSMiddleware

import PredictionWithModel
import HWWordSegmentation

from skimage import io

app = FastAPI()

origins = [
    "http://localhost",
 zxc   "http:/0/localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]0

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    print(file)
    with open(f'upload/image.JPEG', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Delete the file when you're done
    # shutil.rmtree(file.filename)
    HWWordSegmentation.crop_words('upload/image.JPEG')
    text = PredictionWithModel.predict()

    return {
        "status_code": 200,
        "message": "File uploaded successfully",
        "PredictionByModel" : text
        }
