import os
import joblib
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

MODEL_DIR = r"C:\Users\Janhvi Katakdhond\Desktop\Project\working 1.3\models\intent-bert"
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key not found. Set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=GEMINI_API_KEY)

tokenizer = None
model = None
label_encoder = None
gemini_model = None

app = FastAPI(title="Intent Detection + Gemini Response API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryInput(BaseModel):
    text: str

@app.on_event("startup")
def load_resources():
    """Load models and resources when the application starts."""
    global tokenizer, model, label_encoder, gemini_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
        
        label_encoder = joblib.load(ENCODER_PATH)

        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        
        print("All models and resources loaded successfully.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise RuntimeError(f"A required file was not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during resource loading: {e}")
        raise RuntimeError(f"Failed to load application resources: {e}")

def predict_intent(text: str):
    """Predict the intent from the input text using the BERT model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    intent = label_encoder.inverse_transform([pred_id])[0]
    return intent

def query_gemini(user_text: str, detected_intent: str):
    """Query the Gemini model with the user text and detected intent."""
    if gemini_model is None:
        raise HTTPException(status_code=503, detail="Gemini model is not available.")
        
    prompt = f"""
    User query: {user_text}
    Detected intent: {detected_intent}

    Please provide a helpful, natural response using online knowledge.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

@app.post("/predict")
async def predict(query: QueryInput):
    """
    Handles incoming queries, predicts intent, and generates a response.
    """
    if not all([tokenizer, model, label_encoder, gemini_model]):
        raise HTTPException(
            status_code=503,
            detail="Application resources are not fully loaded yet. Please try again in a moment."
        )

    user_input = query.text
    try:
        intent = predict_intent(user_input)
        gemini_answer = query_gemini(user_input, intent)
        
        return {"intent": intent, "response": gemini_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
