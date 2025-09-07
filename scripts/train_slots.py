# scripts/train_slots.py
import json
from pathlib import Path
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from seqeval.metrics import precision_score, recall_score, f1_score

MODEL_NAME = "xlm-roberta-base"
PROC_DIR = Path('../data/processed/hinglish_top')
OUT_DIR = Path('../models/slots-xlmr')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def tokenize_and_align_labels(batch, tokenizer, slot_label_list, max_length=128):
    tokenized_inputs = tokenizer(batch['text'], truncation=True, padding=False, return_offsets_mapping=True, max_length=max_length)
    labels = []
    for i, offsets in enumerate(tokenized_inputs['offset_mapping']):
        slot_tags = batch['slots'][i]
        lab = []
        # if lengths already match we map directly
        if len(offsets) == len(slot_tags):
            for t in slot_tags:
                lab.append(slot_label_list.index(t) if t in slot_label_list else slot_label_list.index('O'))
        else:
            # fallback: map by offset first-char char index into char-level mapping (we don't have char-level here),
            # so choose 'O' for unmatched tokens to avoid errors
            for (s_off, e_off) in offsets:
                if s_off == e_off:
                    lab.append(-100)
                else:
                    lab.append(slot_label_list.index('O'))
        labels.append(lab)
    tokenized_inputs['labels'] = labels
    tokenized_inputs.pop('offset_mapping')
    return tokenized_inputs

def compute_metrics(p, slot_label_list):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)
    true_labels = []; pred_labels = []
    for i, lab in enumerate(labels):
        tlist = []; plist = []
        for j, v in enumerate(lab):
            if v != -100:
                tlist.append(slot_label_list[v])
                plist.append(slot_label_list[preds[i][j]])
        true_labels.append(tlist); pred_labels.append(plist)
    return {"precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels)}

def main():
    ds = load_from_disk(str(PROC_DIR))
    with open('../data/slot2id.json', 'r', encoding='utf-8') as f:
        slot2id = json.load(f)
    slot_label_list = sorted(slot2id.keys(), key=lambda x: slot2id[x])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(slot_label_list))

    def map_fn(batch):
        return tokenize_and_align_labels(batch, tokenizer, slot_label_list)

    tokenized = ds.map(map_fn, batched=True, remove_columns=['text','intent','slots'])
    tokenized.set_format('torch')

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        evaluation_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, slot_label_list)
    )

    trainer.train()
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print("Saved slot model to", OUT_DIR)

if __name__ == "__main__":
    main()
