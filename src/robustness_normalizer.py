import re

def normalize_text(text: str) -> str:
    # 1. Reverse common spacing evasion (e.g., "m e e t" -> "meet")
    # Check if text has excessive character spacing
    if re.search(r'\b\w(\s\w)+\b', text):
        # Collapse single spaced letters while keeping word spaces
        words = text.split("   ")
        normalized_words = []
        for w in words:
            if " " in w and len(w.split()) > 3:
                normalized_words.append("".join(w.split()))
            else:
                normalized_words.append(w)
        text = " ".join(normalized_words)

    # 2. Reverse basic leetspeak substitutions
    leet_map = {'3': 'e', '4': 'a', '7': 't', '0': 'o', '1': 'i', '$': 's'}
    for k, v in leet_map.items():
        text = text.replace(k, v)

    # 3. Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    evaded_sample = "M337 m3 1n $3cr37"
    print("Evaded:", evaded_sample)
    print("Normalized:", normalize_text(evaded_sample))