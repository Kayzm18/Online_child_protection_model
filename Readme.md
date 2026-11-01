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