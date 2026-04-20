import os
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessor import preprocess_text


def rank_resumes(job_description, role, top_n=5, threshold=0.05):

    # Paths
    role_path = f"roles/{role}/processed"
    vector_path = os.path.join(role_path, "X.npy")
    vectorizer_path = os.path.join(role_path, "vectorizer.pkl")
    resume_folder = f"roles/{role}/resumes"

    # Load data
    X = np.load(vector_path)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Preprocess JD
    jd_cleaned = preprocess_text(job_description)

    # Convert JD → vector
    jd_vector = vectorizer.transform([jd_cleaned])

    # Similarity
    similarities = cosine_similarity(jd_vector, X)[0]

    # Sort resume files for correct mapping
    resume_files = sorted(os.listdir(resume_folder))

    results = []

    for resume, score in zip(resume_files, similarities):

        # Threshold filtering
        if score < threshold:
            continue

        results.append({
            "resume": resume,
            "score": round(float(score), 4)
        })

    # Sort by score
    ranked_results = sorted(results, key=lambda x: x["score"], reverse=True)

    return ranked_results[:top_n]