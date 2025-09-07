import os
import joblib
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import requests

# ============================
# CONFIGURATION
# ============================
MODEL_DIR = r"C:\Users\Janhvi Katakdhond\Desktop\Project\working 1.3\models\intent-bert"
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
genai.configure(api_key=GEMINI_API_KEY)

# YouTube Search API (optional – free tier via SerpAPI or custom scraping)
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # set in your env if available

# ============================
# GLOBAL VARS
# ============================
tokenizer, model, label_encoder, gemini_model = None, None, None, None

# ============================
# FASTAPI APP
# ============================
app = FastAPI(title="AI Framework for Real-Time Indian Language and Dialect Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# MODELS
# ============================
class QueryInput(BaseModel):
    text: str
    mode: str = "chat"  # chat | video

@app.on_event("startup")
def load_resources():
    """Load models and resources at startup."""
    global tokenizer, model, label_encoder, gemini_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
        label_encoder = joblib.load(ENCODER_PATH)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ All models loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load resources: {e}")

# ============================
# HELPERS
# ============================
def predict_intent(text: str):
    """Predict user intent with fine-tuned BERT."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([pred_id])[0]

def query_gemini(user_text: str, detected_intent: str):
    """Generate response from Gemini."""
    prompt = f"""
    Project: An AI Framework for Real-Time Indian Language and Dialect Intelligence
    User query: {user_text}
    Detected intent: {detected_intent}

    Provide a clear, concise, and natural response in simple language.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

def fetch_study_videos(query: str):
    """Fetch study videos from YouTube (if API key provided)."""
    if not YOUTUBE_API_KEY:
        return [{"title": "YouTube API key not set", "url": ""}]
    
    params = {
        "part": "snippet",
        "q": query,
        "key": YOUTUBE_API_KEY,
        "maxResults": 3,
        "type": "video"
    }
    try:
        r = requests.get(YOUTUBE_SEARCH_URL, params=params)
        data = r.json()
        videos = [
            {
                "title": item["snippet"]["title"],
                "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            }
            for item in data.get("items", [])
        ]
        return videos
    except Exception as e:
        return [{"title": "Error fetching videos", "url": str(e)}]

# ============================
# API ROUTE
# ============================
@app.post("/predict")
async def predict(query: QueryInput):
    """Main chatbot endpoint."""
    if not all([tokenizer, model, label_encoder, gemini_model]):
        raise HTTPException(status_code=503, detail="Resources not ready yet.")

    try:
        intent = predict_intent(query.text)

        if query.mode == "video":
            videos = fetch_study_videos(query.text)
            return {"intent": intent, "response": "Here are some study videos:", "videos": videos}
        
        else:  # chat mode
            gemini_answer = query_gemini(query.text, intent)
            return {"intent": intent, "response": gemini_answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

# ============================
# RUN SERVER
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
