import os
import joblib
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import requests

# ==========================
# CONFIGURATION
# ==========================
MODEL_DIR = r"C:\Users\Janhvi Katakdhond\Desktop\Project\working 1.3\models\intent-bert"
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY as environment variable.")
genai.configure(api_key=GEMINI_API_KEY)

# YouTube API
YOUTUBE_API_KEY = "AIzaSyDmC-5kMUQBHjSG8q1R66FLkNP3cU9gkKQ"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# ==========================
# GLOBALS
# ==========================
tokenizer = None
model = None
label_encoder = None
gemini_model = None

app = FastAPI(title="Intent Detection + Gemini + YouTube API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# DATA MODEL
# ==========================
class QueryInput(BaseModel):
    text: str

# ==========================
# STARTUP LOAD
# ==========================
@app.on_event("startup")
def load_resources():
    global tokenizer, model, label_encoder, gemini_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
        label_encoder = joblib.load(ENCODER_PATH)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Models + Gemini loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Error loading resources: {e}")

# ==========================
# HELPERS
# ==========================
def predict_intent(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    intent = label_encoder.inverse_transform([pred_id])[0]
    return intent

def query_gemini(user_text: str, detected_intent: str):
    prompt = f"""
    User query: {user_text}
    Detected intent: {detected_intent}

    Please provide a helpful, natural response using online knowledge.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

def search_youtube(query: str, max_results=3):
    params = {
        "part": "snippet",
        "q": query,
        "key": YOUTUBE_API_KEY,
        "type": "video",
        "maxResults": max_results
    }
    resp = requests.get(YOUTUBE_SEARCH_URL, params=params)
    if resp.status_code != 200:
        return []
    data = resp.json()
    results = []
    for item in data.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        results.append({"title": title, "url": url})
    return results

# ==========================
# API ENDPOINT
# ==========================
@app.post("/predict")
async def predict(query: QueryInput):
    if not all([tokenizer, model, label_encoder, gemini_model]):
        raise HTTPException(status_code=503, detail="Resources not ready. Try again shortly.")

    user_input = query.text
    try:
        intent = predict_intent(user_input)

        # 🎵 If intent is music → search YouTube
        if "music" in intent.lower() or "song" in intent.lower():
            yt_results = search_youtube(user_input)
            return {"intent": intent, "response": "Here are some songs I found:", "youtube": yt_results}

        # 📚 If intent is study → search YouTube
        elif "study" in intent.lower() or "learn" in intent.lower():
            yt_results = search_youtube(user_input)
            return {"intent": intent, "response": "Here are some study videos I found:", "youtube": yt_results}

        # 🤖 Otherwise → Gemini answer
        else:
            gemini_answer = query_gemini(user_input, intent)
            return {"intent": intent, "response": gemini_answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
