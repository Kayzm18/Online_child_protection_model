# Child Safety Content Triage

An AI triage system that flags harmful and 18+ content on social media platforms to protect children, adapts as new evasion tactics appear, and keeps human moderators in control of every final decision.

Built for the **Meta Global AI Developer Hackathon** (Oct 26 - Nov 4, 2026).

- **Track:** [Indie Developer / Startup]
- **Team:** 

---

## The Problem

Platforms struggle to catch content that puts children at risk, and bad actors constantly change how they hide it: coded slang, character substitution, emoji codes, cropped or altered images. Static filters fall behind within weeks, and moderators are overwhelmed by volume.

## What We Built

A triage layer that scores incoming text and images, explains why each item was flagged, sends uncertain cases to human reviewers, and learns from their corrections.

1. **Text classifier:** detects grooming and solicitation patterns and sexual content in messages.
2. **Image pipeline:** an adult-content classifier combined with perceptual hash matching (PDQ) to catch re-uploads and tampered copies.
3. **Evasion-robustness layer:** text normalization (leetspeak, homoglyphs, spacing tricks, coded terms) and tamper-resistant image hashing (crops, filters, overlays, compression).
4. **Moderator feedback loop:** low-confidence items go to a review queue, and moderator corrections are used to retrain and improve the models.
5. **Dashboard:** review queue, flagged items with explanations, and before/after evaluation charts.

## Design Principles

- **Human in the loop.** The system prioritizes and flags. People make the final call, especially where children are involved.
- **Explainable flags.** Every flag carries a reason and a confidence score so decisions can be audited.
- **No autonomous online learning.** Models improve only through reviewed, human-approved corrections, which protects against data poisoning.
- **Minimize false positives.** Wrongly flagging a family photo or a teenager's chat does real harm, so we report false-positive rate alongside recall.

## Data Policy

**This project never collects, stores, or trains on illegal material, including child sexual abuse material (CSAM).** No such content is in this repository, and none was used to build or test it.

We use only legal proxy data and synthetic data:

| Purpose | Data |
|---|---|
| Grooming/predator text detection | PAN-12 sexual predator identification dataset |
| Toxicity and harassment text | [Dataset name and license] |
| Adult-content image classification | [Dataset name and license] |
| Benign negatives | [Dataset name and license] |
| Evasion testing | Programmatically generated obfuscations of the datasets above |
| Hash matching demo | Hash list built from harmless images |

Datasets are downloaded locally and are not committed to the repo (`data/` is git-ignored). Check each dataset's license before use.

**Production path:** a real deployment would plug in vetted hash databases (such as those provided to approved organizations by NCMEC, the Internet Watch Foundation, and Thorn), integrate confirmed-material reporting to NCMEC's CyberTipline, and operate under the legal and moderator-wellbeing protocols those organizations require.

## Meta Resources Used

- **Open-weights models:** [model names], used for [fine-tuned text/image classification]
- **Meta Model API (Muse):** [model names], used for [flag explanations, synthetic evasion data generation]
- **SAM 3.1:** [if used] for [flagged-region redaction for moderators]
- **Muse Code:** [if used] for [development]
- **PDQ hashing (ThreatExchange):** perceptual image hashing

*(Update this section with what you actually use before submitting.)*

## Architecture

```
Incoming content
      |
      +-- Text  --> Normalizer --> Text classifier --+
      |                                              +--> Score + explanation
      +-- Image --> PDQ hash match + NSFW classifier-+          |
                                                               v
                                         High confidence --> Flag queue
                                         Low confidence  --> Human review queue
                                                               |
                                                    Moderator corrections
                                                               |
                                                               v
                                                     Retraining / evaluation
```

## Repository Structure

```
.
+-- app/        # API and dashboard
+-- text/       # text classifier, normalizer, training
+-- image/      # image classifier, PDQ hashing
+-- eval/       # metrics, evasion test generators, results
+-- data/       # local datasets (git-ignored)
+-- README.md
```

## Setup

```bash
git clone [repo-url]
cd [repo-name]
pip install -r requirements.txt

# Set your Meta Model API key
export META_API_KEY=your_key_here

# Download datasets (see Data Policy for sources)
python data/download.py

# Run evaluation
python eval/run_eval.py

# Start the app
python app/main.py
```

*(Adjust the commands to match your final code.)*

## Evaluation

We measure precision, recall, F1, and false-positive rate on fixed held-out test splits, and compare performance on clean versus obfuscated inputs.

| Model version | Clean F1 | Obfuscated F1 | False-positive rate |
|---|---|---|---|
| v0 baseline (pretrained, no changes) | [ ] | [ ] | [ ] |
| v1 with robustness layer | [ ] | [ ] | [ ] |
| v2 after moderator feedback retrain | [ ] | [ ] | [ ] |

Results are saved in `eval/`.

## Self-Learning Loop

1. The model scores content, and items below the confidence threshold go to the review queue.
2. A moderator confirms or corrects each decision.
3. Corrections are stored and used to fine-tune the model on a schedule.
4. We re-run the evaluation to verify improvement and check for regressions before accepting a new version.

In the demo we introduce a new evasion pattern, show the baseline missing it, apply moderator corrections, and show the retrained model catching it.

## Limitations

- Trained and evaluated on legal proxy data, so performance on real platform traffic will differ.
- Age-related signals from images are error-prone and are treated only as triage hints.
- The hash-matching demo uses a stand-in list, not official databases.
- Language coverage is limited to [languages].
- Not a replacement for human moderation or legal reporting obligations.

## Responsible Use

This tool is intended for trust and safety teams and child protection organizations. It should be deployed with human review, moderator wellbeing support, privacy safeguards, and compliance with local law. If you encounter suspected child sexual abuse material, do not copy or share it. Report it to the NCMEC CyberTipline (report.cybertip.org) or the Internet Watch Foundation (report.iwf.org.uk).

## License

[Choose a license, e.g., MIT or Apache 2.0]

## Acknowledgments

Meta Global AI Developer Hackathon, Cerebral Valley, and the creators of the open datasets and tools used in this project.