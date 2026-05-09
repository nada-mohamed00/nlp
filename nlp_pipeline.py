import re
import math
from collections import Counter

import spacy
from spacy.matcher import PhraseMatcher
from PyPDF2 import PdfReader
from docx import Document


SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r", "matlab",
    "html", "css", "sass", "tailwind", "bootstrap",
    "react", "angular", "vue", "svelte", "next.js", "nuxt",
    "node.js", "express", "flask", "django", "fastapi", "spring", "laravel",
    "sql", "postgresql", "mysql", "sqlite", "mongodb", "redis", "elasticsearch",
    "docker", "kubernetes", "terraform", "ansible", "jenkins", "github actions",
    "aws", "azure", "gcp", "heroku", "vercel", "netlify",
    "git", "linux", "bash", "rest api", "graphql", "grpc",
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "spacy", "nltk",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "data analysis", "data visualization", "tableau", "power bi", "excel",
    "ci/cd", "agile", "scrum", "jira",
    "communication", "leadership", "teamwork", "problem solving",
]


nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("SKILLS", [nlp.make_doc(s) for s in SKILLS])


def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def read_docx(file):
    document = Document(file)
    return "\n".join(p.text for p in document.paragraphs)


def read_txt(file):
    return file.read().decode("utf-8", errors="ignore")


def parse_resume(file, filename):
    name = filename.lower()
    if name.endswith(".pdf"):
        return read_pdf(file)
    if name.endswith(".docx"):
        return read_docx(file)
    return read_txt(file)


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    doc = nlp(text.lower())
    return [t.lemma_ for t in doc if t.is_alpha and not t.is_stop and len(t.lemma_) > 1]


def extract_name(text):
    doc = nlp(text[:600])
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    return ""


def extract_email(text):
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    return match.group(0) if match else ""


def extract_phone(text):
    match = re.search(r"(\+?\d[\d\s().-]{7,}\d)", text)
    return match.group(0) if match else ""


def extract_skills(text):
    doc = nlp(text.lower())
    found = set()
    for _, start, end in matcher(doc):
        found.add(doc[start:end].text)
    return sorted(found)


def compute_idf(token_lists):
    n = len(token_lists)
    df = Counter()
    for tokens in token_lists:
        for term in set(tokens):
            df[term] += 1
    return {term: math.log((n + 1) / (count + 1)) + 1 for term, count in df.items()}


def tfidf_vector(tokens, idf):
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {term: (count / total) * idf.get(term, 0.0) for term, count in counts.items()}


def cosine(a, b):
    if not a or not b:
        return 0.0
    shared = set(a).intersection(b)
    dot = sum(a[k] * b[k] for k in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_candidates(job_description, resumes):
    job_clean = clean_text(job_description)
    job_tokens = tokenize(job_clean)
    job_skills = set(extract_skills(job_clean))

    parsed = []
    for filename, file in resumes:
        text = clean_text(parse_resume(file, filename))
        parsed.append((filename, text, tokenize(text)))

    idf = compute_idf([job_tokens] + [tokens for _, _, tokens in parsed])
    job_vec = tfidf_vector(job_tokens, idf)

    results = []
    for filename, text, tokens in parsed:
        skills = extract_skills(text)
        matched = sorted(job_skills.intersection(skills)) if job_skills else []
        text_score = cosine(job_vec, tfidf_vector(tokens, idf))
        skill_score = (len(matched) / len(job_skills)) if job_skills else 0.0
        score = 0.7 * text_score + 0.3 * skill_score
        results.append({
            "filename": filename,
            "name": extract_name(text),
            "email": extract_email(text),
            "phone": extract_phone(text),
            "skills": skills,
            "matched_skills": matched,
            "score": round(score * 100, 2),
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results
