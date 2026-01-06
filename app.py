import streamlit as st
import re
import nltk
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="AI Resume–Job Matcher",
    layout="centered"
)
st.title("📄 AI-Based Resume–Job Matching & Skill Gap Analyzer")

# ----------------------------
# Load stopwords safely
# ----------------------------
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
        if page.extract_text():
            text += page.extract_text()
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def extract_skills(text, skill_list):
    found = set()
    for skill in skill_list:
        if skill in text:
            found.add(skill)
    return found

# ----------------------------
# Skill database
# ----------------------------
skill_list = [
    "python", "java", "machine learning", "deep learning", "data science",
    "sql", "mysql", "nlp", "tensorflow", "pytorch", "keras",
    "aws", "azure", "docker", "git",
    "react", "node", "javascript", "html", "css"
]

# ----------------------------
# UI
# ----------------------------
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

if st.button("Analyze Resume"):
    if resume_file is None or job_description.strip() == "":
        st.warning("Please upload resume and job description.")
    else:
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
        st.metric("Resume–Job Match Score", f"{match_score}%")

        st.write("### ✅ Matched Skills")
        st.write(list(matched) if matched else "None")

        st.write("### ❌ Missing Skills")
        st.write(list(missing) if missing else "None")

        st.info(
            "This system uses NLP-based TF-IDF vectorization and cosine similarity, "
            "similar to real Applicant Tracking Systems (ATS)."
        )
        
