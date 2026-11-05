# Child Safety Triage & Evasion-Robustness AI

> **Meta Global AI Developer Hackathon | Track: Indie Developer**  
> *A self-learning triage system that flags online grooming and solicitation risk, neutralizes adversarial evasion attacks, generates structured audit explanations, and keeps human moderators in control.*

---

## Problem & Impact (30% Score Alignment)
Online moderators face an overwhelming volume of content, and predatory bad actors actively use text obfuscation (leetspeak, zero-width spaces, character insertion) to bypass static keyword filters. 

This project provides an adaptive triage layer that:
1. Detects grooming/solicitation patterns early.
2. Normalizes obfuscated text to maintain high recall during active evasion attacks.
3. Provides structured explanations (via Meta AI / Muse Spark concepts) for every flagged conversation.
4. Preserves human oversight through a SQLite-backed reviewer queue and moderator feedback loop.

---

## Architecture & Technical Depth

 Incoming Chat Stream 


[ Evasion-Robustness Normalizer ]  <-- Strips leetspeak, spacing & homoglyphs


[ v1 LoRA Safety Classifier ]      <-- Fine-tuned open-weights sequence classifier


[ Meta AI / Muse Spark Explainer ] <-- Generates structured risk audit trail


[ Moderator Review Queue (SQLite) ] <-- Human confirmation & feedback loop for retraining

---

## Empirical Evaluation & Results

| Model Version | Target Evaluation | Recall | Precision | F1 Score | False Positive Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **v0 Baseline** | Naive Keyword Heuristic | 0.00% | 0.00% | 0.00% | 0.27% |
| **v1 Model (Raw Obfuscated)** | Direct Evasion Attack | 0.00% | 0.00% | 0.00% | 0.00% |
| **v1 Model + Robustness Layer** | Normalized Evasion Defense | **84.62%** | **23.91%** | **37.29%** | **12.19%** |

*Key Technical Finding:* Without our robustness layer, adversarial evasion attacks reduced model recall to 0%. With the normalization layer active, recall jumped to **84.6%**, proving the necessity of pre-classification normalization in trust and safety pipelines.

---

##  Safety, Data & Ethics Policy
1. **Legal Data Usage:** Trained and evaluated strictly using anonymized benchmarks (PAN-12 Sexual Predator Identification Dataset).
2. **Zero CSAM / Explicit Material:** No illegal material or explicit text was produced, collected, or stored anywhere in this repository.
3. **Human-in-the-Loop:** The AI system prioritizes and explains flags; human moderators make all final moderation decisions.
4. **Secret Protection:** All API credentials are securely managed via environment variables (`.env`) and excluded via `.gitignore`.

---

##  Quickstart & Setup

# Clone repo & create environment
git clone [https://github.com/YOUR_USERNAME/child-safety-triage-ai.git](https://github.com/YOUR_USERNAME/child-safety-triage-ai.git)
cd child-safety-triage-ai
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit Dashboard
streamlit run src/app.py