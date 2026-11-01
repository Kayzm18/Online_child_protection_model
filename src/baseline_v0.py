import sys
from pathlib import Path

# Add project root to sys.path so 'eval' can be imported cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import json
from eval.evaluate_baseline import compute_metrics, save_eval_results

def run_baseline():
    test_path = ROOT_DIR / "data/processed/test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Test split not found at {test_path}. Run parser first.")

    y_true = []
    y_pred = []

    print("Running v0 Baseline evaluation on test split...")
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            y_true.append(item["label"])
            
            # v0 Rule-based heuristic baseline for initial sanity check
            text_lower = item["text"].lower()
            predator_keywords = ["meet", "secret", "dont tell", "age", "privately", "webcam"]
            score = sum(1 for kw in predator_keywords if kw in text_lower)
            y_pred.append(1 if score >= 2 else 0)

    metrics = compute_metrics(y_true, y_pred)
    save_eval_results("v0_baseline", metrics)

if __name__ == "__main__":
    run_baseline()