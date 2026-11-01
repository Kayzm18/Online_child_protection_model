import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from peft import get_peft_model, LoraConfig, TaskType

MODEL_NAME = "distilbert-base-uncased"  # Fast, robust encoder backbone for classification
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 1
LR = 2e-5

class ChatDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_length: int):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt")
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }

def train_v1():
    print(f"Loading tokenizer and model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Apply LoRA for parameter-efficient tuning (Meta tooling alignment)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"]
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    train_path = ROOT_DIR / "data/processed/train.jsonl"
    dataset = ChatDataset(train_path, tokenizer, MAX_LENGTH)
    # Use a small subset (e.g., 2000 samples) for fast hackathon training speed
    from torch.utils.data import Subset
    import numpy as np
    subset_indices = np.random.choice(len(dataset), size=min(2000, len(dataset)), replace=False)
    train_subset = Subset(dataset, subset_indices)
    
    loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Training v1 model on {device} for {EPOCHS} epoch(s)...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")

    # Save model
    out_dir = ROOT_DIR / "model_weights/v1_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"v1 LoRA model saved successfully to {out_dir}!")

if __name__ == "__main__":
    train_v1()