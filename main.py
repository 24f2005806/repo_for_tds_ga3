from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

app = FastAPI()

sia = SentimentIntensityAnalyzer()

class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1)

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5)

def analyze(text):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.5:
        return "positive", 5
    elif compound > 0.1:
        return "positive", 4
    elif compound >= -0.1:
        return "neutral", 3
    elif compound > -0.5:
        return "negative", 2
    else:
        return "negative", 1

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(data: CommentRequest):
    try:
        sentiment, rating = analyze(data.comment)
        return {"sentiment": sentiment, "rating": rating}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))