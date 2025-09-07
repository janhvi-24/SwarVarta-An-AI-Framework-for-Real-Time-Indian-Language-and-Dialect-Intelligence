# app.py
import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
import pyttsx3
import uvicorn

# ---------------- CONFIG ----------------
MODEL_DIR = r"C:\Users\Janhvi Katakdhond\Desktop\Project\working 1.3\models\label_encoder.pkl"
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Gemini API key (set in env: setx GEMINI_API_KEY "your_api_key_here")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# ---------------- LOAD TOKENIZER & MODEL ----------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()

# ---------------- LOAD LABEL ENCODER ----------------
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# ---------------- FASTAPI APP ----------------
app = FastAPI(title="Intent Detection + Gemini Response API")

class QueryInput(BaseModel):
    text: str
    response_type: str = "text"  # text or voice

# ---------------- HELPERS ----------------
def predict_intent(text: str):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        logits = model(**encodings).logits
    pred_id = torch.argmax(logits, dim=1).item()
    intent = label_encoder.inverse_transform([pred_id])[0]
    return intent

def query_gemini(user_text: str, detected_intent: str):
    gemini = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"""
    User query: {user_text}
    Detected intent: {detected_intent}

    Please provide a helpful, natural response using online knowledge.
    """
    response = gemini.generate_content(prompt)
    return response.text

def generate_voice_response(text: str, filename="response.wav"):
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

# ---------------- API ROUTE ----------------
@app.post("/predict")
async def predict(query: QueryInput):
    intent = predict_intent(query.text)
    gemini_answer = query_gemini(query.text, intent)

    if query.response_type == "voice":
        filename = generate_voice_response(gemini_answer)
        return {"intent": intent, "response": gemini_answer, "voice_file": filename}
    else:
        return {"intent": intent, "response": gemini_answer}

# ---------------- MAIN ----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
