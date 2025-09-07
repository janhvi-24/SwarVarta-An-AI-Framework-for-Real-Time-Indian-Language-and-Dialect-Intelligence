import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
import joblib

# ---------- Paths ----------
DATA_DIR = r"C:\Users\Janhvi Katakdhond\Desktop\Project\working 1.3\Hinglish-TOP-Dataset\Dataset\Human Annotated Data"
MODEL_DIR = r"C:\Users\Janhvi Katakdhond\Desktop\Project\working 1.3\models\intent-bert"

# ---------- Step 1: Load Data ----------
def load_data(data_dir):
    utterances, intents = [], []
    for file in os.listdir(data_dir):
        if file.endswith(".tsv"):
            path = os.path.join(data_dir, file)
            df = pd.read_csv(path, delimiter="\t")

            # Hinglish utterances
            if "cs_query" in df.columns:
                utterances.extend(df["cs_query"].astype(str).tolist())
            else:
                raise ValueError(f"No 'cs_query' column in {file}")

            # Intents from cs_parse
            if "cs_parse" in df.columns:
                parsed_intents = df["cs_parse"].astype(str).apply(
                    lambda x: x.split()[0].replace("[", "").replace("]", "") if "[" in x else "UNKNOWN"
                )
                intents.extend(parsed_intents.tolist())
            else:
                raise ValueError(f"No 'cs_parse' column in {file}")

    return utterances, intents

# ---------- Step 2: Dataset Class ----------
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ---------- Step 3: Train Function ----------
def train():
    print("📥 Loading dataset...")
    utterances, intents = load_data(DATA_DIR)

    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(intents)
    num_labels = len(label_encoder.classes_)
    print(f"✅ Found {num_labels} unique intents")

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        utterances, labels, test_size=0.2, random_state=42
    )

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Create datasets
    train_dataset = IntentDataset(X_train, y_train, tokenizer)
    val_dataset = IntentDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=num_labels
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * 3  # 3 epochs
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Training loop
    print("🚀 Training started...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Training Loss = {avg_loss:.4f}")

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"📊 Validation Accuracy after Epoch {epoch+1}: {accuracy:.4f}")

    # ---------- Save Model, Tokenizer & Encoder ----------
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("💾 Saving Hugging Face model + tokenizer + encoder...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    print(f"🎉 Training complete! Files saved in {MODEL_DIR}")

if __name__ == "__main__":
    train()
