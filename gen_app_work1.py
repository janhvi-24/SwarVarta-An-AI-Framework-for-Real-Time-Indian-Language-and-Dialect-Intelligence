import os
import joblib
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
import pyttsx3

# =====================
# CONFIGURATION
# =====================
MODEL_DIR = r"C:\Users\Janhvi Katakdhond\Desktop\Project\working 1.3\models\intent-bert"
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Device for model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set Gemini API key as environment variable before running:
# setx GEMINI_API_KEY "your_api_key_here"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key not found. Set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=GEMINI_API_KEY)

# =====================
# LOAD MODEL + ENCODER
# =====================
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    
    # Use joblib instead of pickle
    label_encoder = joblib.load(ENCODER_PATH)

except Exception as e:
    raise RuntimeError(f"Error loading model or label encoder: {e}. Please check your file paths.")

# =====================
# FASTAPI APP
# =====================
app = FastAPI(title="Intent Detection + Gemini Response API")

class QueryInput(BaseModel):
    text: str
    response_type: str = "text"  # text or voice

# =====================
# HELPERS
# =====================
def predict_intent(text: str):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        logits = model(**encodings).logits
    pred_id = torch.argmax(logits, dim=1).item()
    intent = label_encoder.inverse_transform([pred_id])[0]
    return intent

def query_gemini(user_text: str, detected_intent: str):
    gemini_model = genai.GenerativeModel("gemini-pro")
    prompt = f"""
    User query: {user_text}
    Detected intent: {detected_intent}

    Please provide a helpful, natural response using online knowledge.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

def generate_voice_response(text: str, filename="response.wav"):
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

# =====================
# API ROUTE
# =====================
@app.post("/predict")
async def predict(query: QueryInput):
    intent = predict_intent(query.text)
    
    try:
        gemini_answer = query_gemini(query.text, intent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API call failed: {e}")

    if query.response_type == "voice":
        try:
            filename = generate_voice_response(gemini_answer)
            return {"intent": intent, "response": gemini_answer, "voice_file": filename}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Text-to-speech conversion failed: {e}")
    else:
        return {"intent": intent, "response": gemini_answer}

# =====================
# RUN APP
# =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
