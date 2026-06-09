import subprocess
import sys
from typing import Optional

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from textSummarizer.pipeline.prediction import PredictionPipeline


class PredictionRequest(BaseModel):
    text: str


app = FastAPI(title="Text Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["root"])
async def health():
    return {"status": "ok"}


@app.get("/train", tags=["training"])
async def training():
    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return Response(
            f"Training failed.\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
            status_code=500,
        )
    return Response("Training successful !!")


@app.post("/predict", tags=["prediction"])
async def predict_route(
    payload: Optional[PredictionRequest] = Body(default=None),
    text: Optional[str] = Query(default=None),
):
    input_text = text or (payload.text if payload else None)
    if not input_text:
        raise HTTPException(status_code=400, detail="Provide text as a query parameter or JSON body.")

    predictor = PredictionPipeline()
    return {"summary": predictor.predict(input_text)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
