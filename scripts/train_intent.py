import json
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate

# -------------------------------------------------------------------
# Path settings
# -------------------------------------------------------------------
script_dir = Path(__file__).parent
BASE_DIR = script_dir.parent

MODEL_NAME = "xlm-roberta-base"
PROC_DIR = BASE_DIR / 'data' / 'processed' / 'hinglish_top'
OUT_DIR = BASE_DIR / 'models' / 'intent-xlmr'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------
# Preprocess function
# -------------------------------------------------------------------
def preprocess(examples, tokenizer, max_length=128):
    return tokenizer(examples['text'], truncation=True, padding=False, max_length=max_length)

# -------------------------------------------------------------------
# Main training function
# -------------------------------------------------------------------
def main():
    # Load processed dataset
    ds = load_from_disk(str(PROC_DIR))

    # Load intent-to-id mapping
    intent2id_path = BASE_DIR / 'data' / 'processed' / 'intent2id.json'
    if not intent2id_path.exists():
        raise FileNotFoundError(f"Error: '{intent2id_path}' not found. Run preprocess.py first!")

    with open(intent2id_path, 'r', encoding='utf-8') as f:
        intent2id = json.load(f)
    num_labels = len(intent2id)

    print(f"Dataset splits available: {ds.keys()}")
    print(f"Number of labels: {num_labels}")

    # Load tokenizer & model with dropout
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Add a classifier_dropout for a 0.3 dropout rate
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_labels,
        classifier_dropout=0.3
    )

    # Map labels
    def map_labels(batch):
        batch['label'] = [intent2id.get(x, -100) for x in batch['intent']]
        return batch

    # Tokenization
    tokenized = ds.map(map_labels, batched=True)
    tokenized = tokenized.map(lambda x: preprocess(x, tokenizer), batched=True)
    tokenized = tokenized.remove_columns(['intent', 'slots'])
    tokenized.set_format('torch')

    data_collator = DataCollatorWithPadding(tokenizer)

    # Load evaluation metrics using the `evaluate` library
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    # Evaluation metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        
        # Compute metrics
        accuracy_result = accuracy_metric.compute(predictions=preds, references=labels)
        precision_result = precision_metric.compute(predictions=preds, references=labels, average="weighted")
        recall_result = recall_metric.compute(predictions=preds, references=labels, average="weighted")
        f1_result = f1_metric.compute(predictions=preds, references=labels, average="weighted")

        return {
            'accuracy': accuracy_result['accuracy'],
            'precision': precision_result['precision'],
            'recall': recall_result['recall'],
            'f1': f1_result['f1']
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        eval_strategy='epoch',  # Corrected keyword
        save_strategy='epoch',
        num_train_epochs=8,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=3e-5,
        warmup_steps=500,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=torch.cuda.is_available(),
        logging_dir=str(OUT_DIR / "logs"),
        logging_steps=50,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"✅ Training finished. Model saved to {OUT_DIR}")

if __name__ == "__main__":
    main()

