import joblib
import os
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from model import preprocess_review_data
from train import train_and_save_model


class ReviewInput(BaseModel):
    review_content: str
    review_title: str
    review_stars: float


MODEL_PATH = "../data/trained_model.pkl"
TRAIN_FILE_PATH = '../data/train.csv'

# API todo
