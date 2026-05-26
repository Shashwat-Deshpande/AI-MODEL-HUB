import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
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
        "http://127.0.0.1:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN missing in environment variables")

# -----------------------------
# Lightweight Local Models
# -----------------------------

# Sentiment Analysis
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Text Generation
text_generation_pipeline = pipeline(
    "text-generation",
    model="distilgpt2"
)

# -----------------------------
# Request Schema
# -----------------------------

class AIRequest(BaseModel):
    inputs: str

# -----------------------------
# Health Check
# -----------------------------

@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": "Multi-Model AI Server is running flawlessly."
    }

# -----------------------------
# Text Generation Endpoint
# -----------------------------

@app.post("/api/generate")
async def generate_text(request: AIRequest):

    result = text_generation_pipeline(
        request.inputs,
        max_length=50,
        num_return_sequences=1
    )

    return {
        "result": result[0]["generated_text"]
    }

# -----------------------------
# Sentiment Analysis Endpoint
# -----------------------------

@app.post("/api/sentiment")
async def analyze_sentiment(request: AIRequest):

    result = sentiment_pipeline(request.inputs)

    return {
        "result": result[0]
    }

# -----------------------------
# Run Server
# -----------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )