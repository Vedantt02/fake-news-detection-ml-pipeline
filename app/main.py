from fastapi import FastAPI
from pydantic import BaseModel
import os
import pickle

from .utils import preprocess

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "fake_news_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))


# Load FastAPI 
app = FastAPI()

# Request schema
class NewsRequest(BaseModel):
    text: str

# Root Endpoint
@app.get("/")
def home():
    return {"message": "Fake News Detection API is running"}

# Prediction Endpoint
@app.post("/predict")
def predict(request: NewsRequest):
    raw_text = request.text

    # preprocess
    cleaned_text = preprocess(raw_text)

    # vectorize
    vectorized_text = vectorizer.transform([cleaned_text])

    # predict
    prediction = model.predict(vectorized_text)[0]
    probs = model.predict_proba(vectorized_text)[0]

    confidence = max(probs)

    result = "Fake" if prediction == 0 else "Real"

    if confidence < 0.7:
        status = "Uncertain"
    else:
        status = "Confident"

    return {
        "prediction": result,
        "confidence": round(float(confidence), 3),
        "fake_prob": round(float(probs[0]), 3),
        "real_prob": round(float(probs[1]), 3),
        "status": status
    }