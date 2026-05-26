import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from transformers import pipeline

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Multi-Model AI Serving API",
    description="Production-ready API routing requests seamlessly to Hugging Face models."
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing in .env file")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Text generation model
TEXT_GEN_MODEL = "https://api-inference.huggingface.co/models/google/flan-t5-base"

# LOCAL sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

text_generation_pipeline = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

# Request schema
class AIRequest(BaseModel):
    inputs: str

# Hugging Face API helper
def query_hf_api(url: str, payload: dict):
    try:
        response = requests.post(
            url,
            headers=HEADERS,
            json=payload,
            timeout=30
        )

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Hugging Face request timed out."
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text
        )

    return response.json()

@app.post("/api/generate")
async def generate_text(request: AIRequest):

    result = text_generation_pipeline(
        request.inputs,
        max_new_tokens=100
    )

    generated_text = result[0]["generated_text"]

    return {
        "result": generated_text
    }

# Local sentiment analysis endpoint
@app.post("/api/sentiment")
async def analyze_sentiment(request: AIRequest):

    result = sentiment_pipeline(request.inputs)

    return {
        "result": result[0]
    }

# Run server
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )