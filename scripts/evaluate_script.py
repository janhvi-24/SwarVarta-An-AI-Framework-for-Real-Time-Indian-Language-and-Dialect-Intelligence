import json
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding

from sklearn.metrics import classification_report

# Directories setup
script_dir = Path(__file__).parent
BASE_DIR = script_dir.parent  # points to "working 1.3"

MODEL_DIR = BASE_DIR / 'models' / 'intent-xlmr'
PROC_DIR = BASE_DIR / 'data' / 'processed' / 'hinglish_top'
INTENT2ID_PATH = BASE_DIR / 'data' / 'processed' / 'intent2id.json'

def preprocess(examples, tokenizer, max_length=128):
    return tokenizer(examples['text'], truncation=True, padding=False, max_length=max_length)

def main():
    if not MODEL_DIR.exists():
        print(f"Error: Model directory not found at {MODEL_DIR}")
        return

    if not PROC_DIR.exists():
        print(f"Error: Processed dataset not found at {PROC_DIR}")
        return

    if not INTENT2ID_PATH.exists():
        print(f"Error: intent2id.json not found at {INTENT2ID_PATH}")
        return

    # Load mappings
    with open(INTENT2ID_PATH, 'r', encoding='utf-8') as f:
        intent2id = json.load(f)
    id2intent = {v: k for k, v in intent2id.items()}

    # Load dataset and model
    ds = load_from_disk(str(PROC_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    # Map labels (same as in training)
    def map_labels(batch):
        batch['label'] = [intent2id.get(x, -100) for x in batch['intent']]
        return batch

    tokenized = ds.map(map_labels, batched=True)
    
    # ADDED: Tokenize the text column, same as training script
    tokenized = tokenized.map(lambda x: preprocess(x, tokenizer), batched=True)
    
    tokenized = tokenized.remove_columns(['intent', 'slots'])
    tokenized.set_format('torch')

    # Trainer for evaluation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer
    )

    # Run prediction on test set
    predictions = trainer.predict(tokenized['test'])
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # Print metrics
    print("\nEvaluation Results on Test Set:")
    print(classification_report(labels, preds, target_names=[id2intent[i] for i in range(len(id2intent))]))

if __name__ == "__main__":
    main()
