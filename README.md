---
title: HireSmart NLP
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# HireSmart NLP

Resume screening and candidate ranking.

## Stack
- **SpaCy** — tokenization, NER (person names), `PhraseMatcher` for skill extraction
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — semantic similarity between job description and resumes
- **Flask** — recruiter web panel

## Pipeline
Parse → Extract Skills → Similarity Matching → Ranking

Final score = 0.7 × semantic cosine similarity + 0.3 × matched-skill fraction.

## Local run
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python flask_app.py
```
