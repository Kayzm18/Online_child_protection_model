import os
import json
import xml.etree.ElementTree as ET  
import random
from pathlib import Path
from typing import List, Dict, Any

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PREDATOR_TXT = RAW_DATA_DIR / "pan12-sexual-predator-identification-groundtruth-problem1.txt"   
CHAT_XML = RAW_DATA_DIR / "pan12-sexual-predator-identification-test-corpus-2012-05-17.xml"    

WINDOW_SIZE = 15
SEED = 42

def load_predator_ids(path: Path) -> set:
    if not path.exists():
        raise FileNotFoundError(f"Predator IDs file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def parse_and_process():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    predator_ids = load_predator_ids(PREDATOR_TXT)
    print(f"Loaded {len(predator_ids)} predator IDs")

    tree = ET.parse(CHAT_XML)
    root = tree.getroot()

    conversations: Dict[str, List[Dict[str, Any]]] = {}

    author_map: Dict[str, str] = {}

    def get_anon_author(auth_id: str) -> str:
        if auth_id not in author_map:
            anon_id = f"anon_{len(author_map) + 1}"
            author_map[auth_id] = anon_id
        return author_map[auth_id]

    #process XML messages
    for conversation in root.findall("conversation"):
        conv_id = conversation.get("id", "unknown")
        messages = []
        is_predator_conv = False

        for msg in conversation.findall("message"):
            author = msg.find("author")
            text = msg.find("text")

            auth_id = author.text if author is not None and author.text else "unknown"
            msg_text = text.text if text is not None and text.text else ""

            if auth_id in predator_ids:
                is_predator_conv = True

            messages.append({
                "author": get_anon_author(auth_id),
                "is_predator": auth_id in predator_ids,
                "text": msg_text
                })

        if messages:
            conversations[conv_id] = messages
    print(f"Parsed {len(conversations)} conversations from XML")

    # chunk windows
    dataset_windows = []
    for conv_id, msgs in conversations.items():
        has_predator = any(m["is_predator"] for m in msgs)
        label = 1 if has_predator else 0

        for i in range(0, len(msgs), WINDOW_SIZE):
            window = msgs[i:i + WINDOW_SIZE]
            if len(window) < 3:
                continue

            formatted_text = "\n".join([f"{m['author']}: {m['text']}" for m in window])
            dataset_windows.append({
                "conv_id": conv_id,
                "window_idx": i // WINDOW_SIZE,
                "text": formatted_text,
                "label": label  
                })

        unique_conv_ids = list(set(item["conv_id"] for item in dataset_windows))
        random.seed(SEED)
        random.shuffle(unique_conv_ids)

        n_total = len(unique_conv_ids)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)

        train_ids = set(unique_conv_ids[:n_train])
        val_ids = set(unique_conv_ids[n_train:n_train + n_val])
        test_ids = set(unique_conv_ids[n_train + n_val:])

        splits = {"train": [], "val": [], "test": []}
        for item in dataset_windows:
            if item["conv_id"] in train_ids:
                splits["train"].append(item)
            elif item["conv_id"] in val_ids:
                splits["val"].append(item)
            else:
                splits["test"].append(item)

        for split_name, items in splits.items():
            out_file = PROCESSED_DATA_DIR / f"{split_name}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n") 

            pos_count = sum(1 for x in items if x["label"] == 1)
            neg_count = sum(1 for x in items if x["label"] == 0)
            print(f"Split [{split_name}]: {len(items)} windows | Positive (Predator): {pos_count} | Negative: {neg_count}")

if __name__ == "__main__":
    parse_and_process()


