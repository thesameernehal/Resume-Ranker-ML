import os
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessor import preprocess_text


def rank_resumes(job_description, role, top_n=5):
    """
    Rank resumes based on job description

    Args:
        job_description (str)
        role (str) → accountant / chef / information_technology
        top_n (int)

    Returns:
        List of (resume_name, score)
    """

    role_path = f"roles/{role}/processed"
    vector_path = os.path.join(role_path, "X.npy")
    vectorizer_path = os.path.join(role_path, "vectorizer.pkl")

    resume_folder = f"roles/{role}/resumes"

    # Load data
    X = np.load(vector_path)
    
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Preprocess job description
    jd_cleaned = preprocess_text(job_description)

    # Convert JD → vector
    jd_vector = vectorizer.transform([jd_cleaned])

    # Compute similarity
    similarities = cosine_similarity(jd_vector, X)[0]

    # Get resume names
    resume_files = os.listdir(resume_folder)

    # Combine names + scores
    results = list(zip(resume_files, similarities))

    # Sort (highest score first)
    ranked_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Return top N
    return ranked_results[:top_n]