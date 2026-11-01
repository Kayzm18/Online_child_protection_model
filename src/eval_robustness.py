import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from src.robustness_normalizer import normalize_text
from src.evasion_generator import apply_evasion
from eval.evaluate_baseline import compute_metrics, save_eval_results

def evaluate_robustness():
    model_path = ROOT_DIR / "model_weights/v1_classifier"
    test_path = ROOT_DIR / "data/processed/test.jsonl"

    print("Loading v1 LoRA classifier for robustness evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    model = PeftModel.from_pretrained(base_model, model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true = []
    y_pred_obfuscated_raw = []
    y_pred_normalized = []

    print("Evaluating on test subset with evasion attacks...")
    count = 0
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= 300: # Evaluate on 300 samples for fast hackathon turnaround
                break
            item = json.loads(line)
            y_true.append(item["label"])
            raw_text = item["text"]

            # Apply evasion attack
            evaded_text = apply_evasion(raw_text)

            # 1. Prediction without normalization (Before)
            inputs_raw = tokenizer(evaded_text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits_raw = model(**inputs_raw).logits
                pred_raw = torch.argmax(logits_raw, dim=-1).item()
            y_pred_obfuscated_raw.append(pred_raw)

            # 2. Prediction with robustness normalization layer (After)
            norm_text = normalize_text(evaded_text)
            inputs_norm = tokenizer(norm_text, return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                logits_norm = model(**inputs_norm).logits
                pred_norm = torch.argmax(logits_norm, dim=-1).item()
            y_pred_normalized.append(pred_norm)

            count += 1

    metrics_raw = compute_metrics(y_true, y_pred_obfuscated_raw)
    metrics_norm = compute_metrics(y_true, y_pred_normalized)

    save_eval_results("v1_obfuscated_raw", metrics_raw)
    save_eval_results("v1_robustness_normalized", metrics_norm)

if __name__ == "__main__":
    evaluate_robustness()