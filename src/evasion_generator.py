import random

def apply_leetspeak(text: str) -> str:
    leet_map = {'e': '3', 'a': '4', 't': '7', 'o': '0', 'i': '1', 's': '$'}
    return "".join(leet_map.get(c.lower(), c) for c in text)

def apply_spacing(text: str) -> str:
    # Add extra spaces between letters to bypass simple token/word filters
    return " ".join(list(text))

def apply_evasion(text: str) -> str:
    # Apply random evasion transforms
    choices = [apply_leetspeak, apply_spacing]
    transform = random.choice(choices)
    return transform(text)

if __name__ == "__main__":
    sample = "Meet me in secret and do not tell anyone"
    print("Original:", sample)
    print("Evaded (Leet):", apply_leetspeak(sample))
    print("Evaded (Spacing):", apply_spacing(sample))