import json
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns # Import seaborn

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
    
    # Tokenize the text column, same as training script
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
    
    # --- Plotting and Reporting Section ---
    y_true = predictions.label_ids   # true labels
    y_pred = np.argmax(predictions.predictions, axis=-1)  # predicted labels
    
    # Get label names from the id2intent mapping created earlier
    label_names = [id2intent[i] for i in range(len(id2intent))]
    
    # 1. Classification report dictionary
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=label_names)) # Print the readable report

    # ---- Plot Macro vs Weighted Averages ----
    metrics = ["precision", "recall", "f1-score"]
    macro_scores = [report["macro avg"][m] for m in metrics]
    weighted_scores = [report["weighted avg"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, macro_scores, width, label='Macro Avg')
    plt.bar(x + width/2, weighted_scores, width, label='Weighted Avg')

    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Macro vs Weighted Metrics")
    plt.legend()
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.show()

    # ---- Plot Per-class F1 Scores ----
    f1_scores = [report[label]["f1-score"] for label in label_names]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=label_names, y=f1_scores, color="skyblue") # Pass label_names directly
    plt.xticks(rotation=90, fontsize=8) # Rotate and adjust font size for readability
    plt.ylabel("F1 Score")
    plt.title("Per-class F1 Scores")
    plt.tight_layout() # Adjust layout
    plt.show()

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    
    plt.figure(figsize=(15, 12)) # Increase figure size for better readability with many classes
    sns.heatmap(cm, annot=False, cmap="Blues", 
                xticklabels=label_names, yticklabels=label_names, 
                fmt="d", linewidths=.5, linecolor='gray') # Add lines for better cell separation
    plt.xlabel("Predicted Label", fontsize=12) # Add fontsize for clarity
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=90, fontsize=8) # Rotate and adjust font size
    plt.yticks(rotation=0, fontsize=8) # Keep y-labels horizontal
    plt.tight_layout()
    plt.show()

    # ---- Save Report as JSON ----
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Classification report saved to classification_report.json")
    # --- End Plotting and Reporting Section ---

if __name__ == "__main__":
    main()
