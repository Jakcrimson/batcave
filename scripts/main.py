# Install dependencies before running this code
# !pip install fastapi uvicorn pydantic

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

import train

app = FastAPI()

class TrainRequest(BaseModel):
    file_path: str = Field(..., example="/path/to/train_dataset.csv")


class TestRequest(BaseModel):
    file_path: str = Field(..., example="/path/to/test_dataset.csv")


class UseRequest(BaseModel):
    text1: str = Field(..., example="Sample text input 1")
    text2: str = Field(..., example="Sample text input 2")
    number: int = Field(..., ge=1, le=5, example=3)


@app.post("/train", summary="Train the ML Model")
async def train_model(request: TrainRequest):

    try:
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=500, detail="File not found")

        return_str = train.train_and_save_model(request.file_path)

        return return_str

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test", summary="Test the ML Model")
async def test_model(request: TestRequest):
    try:
        return "Not implemented yet :3"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/use", summary="Use the ML Model for Prediction")
async def use_model(request: UseRequest):
    try:
        return "Not implemented yet :3"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the app, use the command: uvicorn <filename_without_extension>:app --reload
