import re

import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util
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
    "huggingface", "sentence transformers", "openai", "langchain",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly",
    "data analysis", "data visualization", "tableau", "power bi", "excel",
    "ci/cd", "agile", "scrum", "jira",
    "communication", "leadership", "teamwork", "problem solving",
]


nlp = spacy.load("en_core_web_sm")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("SKILLS", [nlp.make_doc(s) for s in SKILLS])
model = SentenceTransformer("all-MiniLM-L6-v2")


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


def embed(texts):
    return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)


def rank_candidates(job_description, resumes):
    job_clean = clean_text(job_description)
    job_skills = set(extract_skills(job_clean))

    parsed = []
    for filename, file in resumes:
        text = clean_text(parse_resume(file, filename))
        parsed.append((filename, text))

    texts = [job_clean] + [text for _, text in parsed]
    vectors = embed(texts)
    job_vec = vectors[0]

    results = []
    for (filename, text), vec in zip(parsed, vectors[1:]):
        skills = extract_skills(text)
        matched = sorted(job_skills.intersection(skills)) if job_skills else []
        text_score = float(util.cos_sim(job_vec, vec)[0][0])
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
