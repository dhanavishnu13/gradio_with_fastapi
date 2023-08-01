# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from main import predict

app = FastAPI()


class Request(BaseModel):
    question: str


class Result(BaseModel):
    score: float
    title: str
    text: str


class Response(BaseModel):
    results: List[Result] # list of Result objects


@app.post("/predict", response_model=Response)
async def predict_api(request: Request):
    results = predict(request.question)
    return Response(
        results=[
            Result(score=r["score"], title=r["title"], text=r["text"])
            for r in results
        ]
    )