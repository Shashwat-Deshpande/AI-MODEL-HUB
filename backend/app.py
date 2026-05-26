import os
import uvicorn
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# Small hosted models
TEXT_MODEL = "https://api-inference.huggingface.co/models/distilgpt2"

SENTIMENT_MODEL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

class AIRequest(BaseModel):
    inputs: str

@app.get("/")
def home():
    return {"status": "online"}

def query_model(url, payload):

    response = requests.post(
        url,
        headers=HEADERS,
        json=payload
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.text
        )

    return response.json()

@app.post("/api/generate")
async def generate_text(request: AIRequest):

    payload = {
        "inputs": request.inputs
    }

    result = query_model(TEXT_MODEL, payload)

    return {
        "result": result[0]["generated_text"]
    }

@app.post("/api/sentiment")
async def sentiment(request: AIRequest):

    payload = {
        "inputs": request.inputs
    }

    result = query_model(SENTIMENT_MODEL, payload)

    return {
        "result": result[0]
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )