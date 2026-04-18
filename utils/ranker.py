import os
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessor import preprocess_text


def extract_keywords(text):
    """
    Simple keyword extraction (basic approach)
    """
    words = text.split()
    return set(words)


def rank_resumes(job_description, role, top_n=5, threshold=0.05):
    """
    Rank and filter resumes
    """

    # 📌 Paths
    role_path = f"roles/{role}/processed"
    vector_path = os.path.join(role_path, "X.npy")
    vectorizer_path = os.path.join(role_path, "vectorizer.pkl")

    resume_folder = f"roles/{role}/resumes"

    # 📌 Load data
    X = np.load(vector_path)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # 📌 Preprocess JD
    jd_cleaned = preprocess_text(job_description)

    # 📌 Extract keywords from JD
    jd_keywords = extract_keywords(jd_cleaned)

    # 📌 Convert JD → vector
    jd_vector = vectorizer.transform([jd_cleaned])

    # 📌 Compute similarity
    similarities = cosine_similarity(jd_vector, X)[0]

    # 📌 Get resume names
    resume_files = os.listdir(resume_folder)

    results = []

    # 📌 Apply filtering
    for resume, score in zip(resume_files, similarities):

        # 🔹 Filter 1: Threshold
        if score < threshold:
            continue

        # 🔹 Filter 2: Keyword match (basic check)
        resume_text_path = os.path.join(role_path, resume + ".txt")

        keyword_match = 0

        # NOTE: For now we skip file reading (optional improvement later)
        # So we just accept threshold-based filtering

        results.append((resume, score))

    # 📌 Sort results
    ranked_results = sorted(results, key=lambda x: x[1], reverse=True)

    return ranked_results[:top_n]