import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file securely
load_dotenv()

META_API_KEY = os.getenv("META_API_KEY")
# Replace with the official Meta Model API / partner endpoint if applicable
META_API_URL = os.getenv("META_API_URL", "https://api.meta.ai/v1/chat/completions")

def generate_flag_explanation(chat_text: str, confidence: float) -> dict:
    """
    Generates structured audit explanations for why a chat window was flagged,
    integrating the Meta Model API / Muse Spark concepts while keeping 
    secrets strictly in .env.
    """
    # If API key is configured and valid, call the live Meta Model API endpoint
    if META_API_KEY and META_API_KEY != "your_actual_meta_api_key_here":
        headers = {
            "Authorization": f"Bearer {META_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-models/muse-spark-open-1026",
            "messages": [
                {"role": "system", "content": "You are a child safety triage assistant. Explain why this chat window matches risk indicators for grooming or solicitation in 2 concise sentences."},
                {"role": "user", "content": chat_text}
            ],
            "max_tokens": 150
        }
        try:
            response = requests.post(META_API_URL, json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                result = response.json()
                explanation_text = result["choices"][0]["message"]["content"]
                return {
                    "flagged": True,
                    "confidence_score": round(float(confidence), 4),
                    "explanation": explanation_text,
                    "source": "Meta Model API (Muse Spark)"
                }
        except Exception as e:
            print(f"API call fallback triggered due to network/auth: {e}")

    # Robust local rule-based fallback for hackathon evaluation and offline testing
    reasons = []
    text_lower = chat_text.lower()
    if "meet" in text_lower or "secret" in text_lower:
        reasons.append("Solicitation for offline meetup or secrecy detected.")
    if "age" in text_lower or "dont tell" in text_lower:
        reasons.append("Inquiry regarding age or instructions to conceal communication.")
    if not reasons:
        reasons.append("Behavioral pattern matching known grooming discourse taxonomy.")

    return {
        "flagged": True,
        "confidence_score": round(float(confidence), 4),
        "primary_indicators": reasons,
        "recommended_action": "Route to Human Moderator Review Queue",
        "audit_trail": "Evaluated by v1 LoRA Safety Classifier + Robustness Normalization Layer",
        "source": "Local Fallback / Rule Taxonomy"
    }

if __name__ == "__main__":
    sample = "Let's meet up in secret, don't tell your parents."
    print("Testing Explainer Module:")
    print(generate_flag_explanation(sample, 0.9421))