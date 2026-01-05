import streamlit as st
import spacy
import re
import nltk
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI Resume–Job Matcher", layout="centered")
st.title("📄 AI-Based Resume–Job Matching & Skill Gap Analyzer")

# ----------------------------
# Load NLP resources
# ----------------------------
@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

nlp = load_nlp()
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

stop_words = load_stopwords()


# ----------------------------
# Helper functions
# ----------------------------
def extract_text_from_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

def extract_skills(text, skill_list):
    doc = nlp(text)
    return set([token.text.lower() for token in doc if token.text.lower() in skill_list])

# ----------------------------
# Skill database
# ----------------------------
skill_list = [
    "python","java","machine learning","deep learning","data science",
    "sql","mysql","nlp","tensorflow","pytorch","keras",
    "aws","azure","docker","git","react","node","javascript","html","css"
]

# ----------------------------
# UI
# ----------------------------
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

if st.button("Analyze Resume") and resume_file and job_description:

    resume_text = extract_text_from_pdf(resume_file)
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_description)

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_clean, job_clean])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    match_score = round(similarity * 100, 2)

    resume_skills = extract_skills(resume_clean, skill_list)
    job_skills = extract_skills(job_clean, skill_list)

    matched = resume_skills & job_skills
    missing = job_skills - resume_skills

    st.subheader("📊 Results")
    st.metric("Match Score", f"{match_score}%")

    st.write("### ✅ Matched Skills")
    st.write(list(matched) if matched else "None")

    st.write("### ❌ Missing Skills")
    st.write(list(missing) if missing else "None")

    st.info("This system uses NLP (TF-IDF + Cosine Similarity) similar to real ATS platforms.")
