# scripts/data_prep_local.py
import csv
import json
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import pandas as pd
import re
import sys

# === EDITABLE: set your local folder containing the TSV files ===
LOCAL_DATA_DIR = Path(r"C:\Users\Janhvi Katakdhond\Desktop\Project\working 1.3\Hinglish-TOP-Dataset\Dataset\Human Annotated Data")
# === output processed data dir (relative to project root) ===
PROC_DIR = Path("../data/processed/hinglish_top")
PROC_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "xlm-roberta-base"

def read_tsv_file(path):
    # The repository's TSV has 5 columns:
    # english_query, code-switched_query, english_parse, code-switched_parse, domain
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for r in reader:
            if not r or all(not c.strip() for c in r):
                continue
            # pad to at least 5 columns
            while len(r) < 5:
                r.append('')
            en_q, cs_q, en_parse, cs_parse, domain = r[:5]
            rows.append({'text': cs_q.strip(), 'parse': cs_parse.strip(), 'domain': domain.strip()})
    return rows

# heuristic extractor — robust for the TOP/parse style in this repo
def extract_slots_from_parse(parse_str, utterance):
    if not parse_str or not isinstance(parse_str, str):
        return []
    slots = []
    # pattern 1: (SLOTNAME "value") or SLOTNAME:"value" or SLOTNAME='value'
    for match in re.finditer(r'([\w:-]+)\s*[:]?[\s]*[\"\']([^\"\']+)[\"\']', parse_str):
        slot_key = match.group(1)
        val = match.group(2)
        ent = slot_key.split(':')[-1].strip()
        # find in utterance (first occurrence, case-insensitive fallback)
        idx = utterance.find(val)
        if idx >= 0:
            slots.append({'entity': ent, 'value': val, 'start': idx, 'end': idx+len(val)})
        else:
            m = re.search(re.escape(val), utterance, flags=re.I)
            if m:
                slots.append({'entity': ent, 'value': utterance[m.start():m.end()], 'start': m.start(), 'end': m.end()})
            else:
                slots.append({'entity': ent, 'value': val, 'start': -1, 'end': -1})
    # pattern 2: slotName[value] or slotName(value)
    for match in re.finditer(r'([\w:-]+)\s*[\[\(]\s*([^\]\)]+)\s*[\]\)]', parse_str):
        slot_key = match.group(1)
        val = match.group(2).strip()
        ent = slot_key.split(':')[-1].strip()
        idx = utterance.find(val)
        if idx >= 0:
            slots.append({'entity': ent, 'value': val, 'start': idx, 'end': idx+len(val)})
        else:
            m = re.search(re.escape(val), utterance, flags=re.I)
            if m:
                slots.append({'entity': ent, 'value': utterance[m.start():m.end()], 'start': m.start(), 'end': m.end()})
            else:
                slots.append({'entity': ent, 'value': val, 'start': -1, 'end': -1})
    # deduplicate
    uniq = {}
    for s in slots:
        k = (s['entity'], s['value'])
        if k not in uniq:
            uniq[k] = s
    return list(uniq.values())

def build_char_tags(text, slot_list):
    tags = ['O'] * len(text)
    for s in slot_list:
        start = s.get('start', -1)
        end = s.get('end', -1)
        val = s.get('value', '')
        ent = s.get('entity', 'UNKNOWN')
        if start == -1 or end == -1:
            if val:
                m = re.search(re.escape(val), text, flags=re.I)
                if m:
                    start, end = m.start(), m.end()
        if start is None or end is None or start < 0 or end <= start:
            continue
        start = max(0, int(start)); end = min(len(text), int(end))
        tags[start] = 'B-' + ent
        for i in range(start+1, end):
            tags[i] = 'I-' + ent
    return tags

def char_to_token_tags(texts, char_tags_list, tokenizer, max_length=128):
    token_tags = []
    for text, char_tags in zip(texts, char_tags_list):
        enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
        offsets = enc['offset_mapping']
        ttags = []
        for (s_off, e_off) in offsets:
            if s_off == e_off:
                ttags.append('O')
            else:
                tag = char_tags[s_off] if s_off < len(char_tags) else 'O'
                ttags.append(tag)
        token_tags.append(ttags)
    return token_tags

def main():
    # locate files
    train_f = LOCAL_DATA_DIR / "train.tsv"
    dev_f = LOCAL_DATA_DIR / "validate.tsv" if (LOCAL_DATA_DIR / "validate.tsv").exists() else LOCAL_DATA_DIR / "dev.tsv"
    test_f = LOCAL_DATA_DIR / "test.tsv"

    available = {}
    if train_f.exists(): available['train'] = read_tsv_file(train_f)
    if dev_f and dev_f.exists(): available['validation'] = read_tsv_file(dev_f)
    if test_f.exists(): available['test'] = read_tsv_file(test_f)

    if not available:
        print("No TSV files found in", LOCAL_DATA_DIR)
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    processed = {}
    intent_set = set(); entity_set = set()

    for split, rows in available.items():
        texts = []; intents = []; char_tags_all = []
        for r in rows:
            text = r['text']
            parse = r['parse']
            domain = r['domain'] or ""
            slots = extract_slots_from_parse(parse, text)
            char_tags = build_char_tags(text, slots)
            texts.append(text)
            # extract intent: look for 'IN:' tokens in parse; else domain; else 'unknown'
            intent = None
            m = re.search(r'IN[:=]?([A-Za-z0-9_/-]+)', parse) if parse else None
            if m:
                intent = m.group(1)
            else:
                m2 = re.search(r'\(([A-Z_]+)\s', parse) if parse else None
                if m2:
                    intent = m2.group(1)
                else:
                    intent = domain or "unknown"
            intents.append(intent)
            intent_set.add(intent)
            for tag in set([t for t in char_tags if t != 'O']):
                if tag.startswith('B-') or tag.startswith('I-'):
                    entity_set.add(tag.split('-', 1)[1])
            char_tags_all.append(char_tags)
        token_tags = char_to_token_tags(texts, char_tags_all, tokenizer)
        processed[split] = pd.DataFrame({'text': texts, 'intent': intents, 'slots': token_tags})

    intent_list = sorted(list(intent_set))
    slot_labels = ['O']
    for e in sorted(entity_set):
        slot_labels.append('B-'+e); slot_labels.append('I-'+e)
    intent2id = {l:i for i,l in enumerate(intent_list)}
    slot2id = {l:i for i,l in enumerate(slot_labels)}

    ds_dict = {}
    for k, df in processed.items():
        ds_dict[k] = Dataset.from_pandas(df)
    dataset = DatasetDict(ds_dict)
    dataset.save_to_disk(str(PROC_DIR))

    with open(PROC_DIR.parent / 'intent2id.json', 'w', encoding='utf-8') as f:
        json.dump(intent2id, f, ensure_ascii=False, indent=2)
    with open(PROC_DIR.parent / 'slot2id.json', 'w', encoding='utf-8') as f:
        json.dump(slot2id, f, ensure_ascii=False, indent=2)

    print("Saved processed dataset to", PROC_DIR)
    print("Intents:", len(intent_list), "Slot labels:", len(slot_labels))

if __name__ == "__main__":
    main()
