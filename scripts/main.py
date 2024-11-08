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
    review_content: str = Field(..., example="review_content")
    review_title: str = Field(..., example="review_title")
    review_stars: int = Field(..., ge=1, le=5, example=3)
    product: str = Field(..., example="product")


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
        if not os.path.exists(request.file_path):
            raise HTTPException(status_code=500, detail="File not found")

        return_str = train.test_best_model(request.file_path)

        if not return_str:
            raise HTTPException(status_code=500, detail="Model not found (have you trained it?)")

        return {"score": return_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", summary="Predict with the ML Model")
async def predict_model(request: UseRequest):
    try:
        print(request.review_content)
        print(request.review_title)
        print(request.review_stars)
        print(request.product)

        return_str = train.use_best_model(request.review_content,
                                          request.review_title,
                                          request.review_stars,
                                          request.product)

        if not return_str:
            raise HTTPException(status_code=500, detail="Model not found (have you trained it?)")

        return {"Prediction": return_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the app, use the command: uvicorn main:app --reload
