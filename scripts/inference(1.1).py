# scripts/inference.py
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import torch

# Get the absolute path of the current script's directory
script_dir = Path(__file__).parent

# Construct the paths relative to the script's directory
INTENT_MODEL = script_dir.parent / 'models' / 'intent-xlmr'
SLOT_MODEL   = script_dir.parent / 'models' / 'slots-xlmr'

# --- The change is here, adding 'processed' to the path ---
INTENT_DATA  = script_dir.parent / 'data' / 'processed' / 'intent2id.json'
SLOT_DATA    = script_dir.parent / 'data' / 'processed' / 'slot2id.json'
# --- End of change ---

with open(INTENT_DATA) as f: intent2id = json.load(f)
id2intent = {v:k for k,v in intent2id.items()}
with open(SLOT_DATA) as f: slot2id = json.load(f)
slot_label_list = sorted(slot2id.keys(), key=lambda x: slot2id[x])

try:
    tokenizer_intent = AutoTokenizer.from_pretrained(str(INTENT_MODEL))
    model_intent = AutoModelForSequenceClassification.from_pretrained(str(INTENT_MODEL)).eval()
except Exception:
    tokenizer_intent = AutoTokenizer.from_pretrained('xlm-roberta-base')
    model_intent = None

try:
    tokenizer_slot = AutoTokenizer.from_pretrained(str(SLOT_MODEL))
    model_slot = AutoModelForTokenClassification.from_pretrained(str(SLOT_MODEL)).eval()
except Exception:
    tokenizer_slot = AutoTokenizer.from_pretrained('xlm-roberta-base')
    model_slot = None

def predict(utterance):
    result = {'intent': None, 'slots': []}
    if model_intent:
        enc = tokenizer_intent(utterance, return_tensors='pt', truncation=True)
        with torch.no_grad():
            out = model_intent(**{k:v.to(model_intent.device) for k,v in enc.items()})
        pred = int(out.logits.argmax(-1).item())
        if hasattr(model_intent.config, 'id2label') and model_intent.config.id2label:
            result['intent'] = model_intent.config.id2label.get(str(pred), id2intent.get(pred))
        else:
            result['intent'] = id2intent.get(pred)
    if model_slot:
        enc2 = tokenizer_slot(utterance, return_offsets_mapping=True, return_tensors='pt', truncation=True)
        offsets = enc2.pop('offset_mapping').squeeze().tolist()
        with torch.no_grad():
            out2 = model_slot(**{k:v.to(model_slot.device) for k,v in enc2.items()})
        preds = out2.logits.argmax(-1).squeeze().cpu().numpy().tolist()
        spans=[]; cur=None
        for p, off in zip(preds, offsets):
            s_off, e_off = off
            if s_off == e_off: continue
            lab = slot_label_list[p]
            if lab == 'O':
                if cur: spans.append(cur); cur=None
                continue
            if lab.startswith('B-'):
                if cur: spans.append(cur)
                cur = [s_off, e_off, lab[2:]]
            elif lab.startswith('I-'):
                if cur and cur[2] == lab[2:]:
                    cur[1] = e_off
                else:
                    cur = [s_off, e_off, lab[2:]]
        if cur: spans.append(cur)
        result['slots'] = [{'entity': s[2], 'value': utterance[s[0]:s[1]], 'start': s[0], 'end': s[1]} for s in spans]
    return result

if __name__ == '__main__':
    while True:
        t = input("Utterance> ")
        if not t: break
        print(predict(t))
