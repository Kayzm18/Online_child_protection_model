import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3  # Increased to 3 epochs for proper convergence
LR = 3e-5   # Slightly optimized learning rate

class ChatDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, max_length: int):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
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
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }

def train_v1():
    print(f"Loading tokenizer and model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

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

    # Create a balanced subset (all available positives + equal negatives for robust learning)
    pos_indices = [i for i, item in enumerate(dataset.data) if item["label"] == 1]
    neg_indices = [i for i, item in enumerate(dataset.data) if item["label"] == 0]
    
    np.random.seed(42)
    selected_pos = np.random.choice(pos_indices, size=min(len(pos_indices), 1000), replace=False)
    selected_neg = np.random.choice(neg_indices, size=min(len(neg_indices), 1000), replace=False)
    balanced_indices = np.concatenate([selected_pos, selected_neg])
    np.random.shuffle(balanced_indices)

    train_subset = Subset(dataset, balanced_indices.tolist())
    loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Training v1 balanced model on {device} for {EPOCHS} epochs ({len(balanced_indices)} samples)...")
    for epoch in range(EPOCHS):
        model.train()
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
        print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss / len(loader):.4f}")

    out_dir = ROOT_DIR / "model_weights/v1_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"v1 Balanced LoRA model saved successfully to {out_dir}!")

if __name__ == "__main__":
    train_v1()