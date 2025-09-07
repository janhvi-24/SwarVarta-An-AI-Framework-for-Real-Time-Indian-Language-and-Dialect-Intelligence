# scripts/inference.py
import torch
import joblib
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification

# Paths
script_dir = Path(__file__).parent
MODEL_DIR = script_dir.parent / "models"

# Load tokenizer, model, and label encoder
print("📥 Loading model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(str(MODEL_DIR))
model = BertForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval()
label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_intent(utterance: str) -> str:
    """Predicts the intent of a given utterance."""
    enc = tokenizer(utterance, return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        outputs = model(**enc)
    pred_id = torch.argmax(outputs.logits, dim=-1).item()

    # Decode intent label
    intent = label_encoder.inverse_transform([pred_id])[0]
    return intent

if __name__ == "__main__":
    print("🚀 Inference ready! Type a sentence (press Enter to quit):")
    while True:
        text = input("Utterance> ").strip()
        if not text:
            break
        intent = predict_intent(text)
        print(f"Predicted Intent: {intent}")
