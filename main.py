from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from transformers import pipeline

app = FastAPI()

# Load sentiment model once (fast for multiple requests)
sentiment_pipeline = pipeline("sentiment-analysis")

# -------- Request Model --------
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

# -------- Response Model --------
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5)

def map_sentiment(label: str, score: float):
    if label == "POSITIVE":
        rating = round(3 + score * 2)
        return "positive", min(rating, 5)
    elif label == "NEGATIVE":
        rating = round(3 - score * 2)
        return "negative", max(rating, 1)
    else:
        return "neutral", 3

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(data: CommentRequest):
    try:
        result = sentiment_pipeline(data.comment)[0]
        sentiment, rating = map_sentiment(result["label"], result["score"])

        return {
            "sentiment": sentiment,
            "rating": rating
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))