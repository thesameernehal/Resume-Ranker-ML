import os
import numpy as np
import pickle
from utils.parser import extract_text
from utils.preprocessor import preprocess_text
from utils.vectorizer import create_tfidf_vectors


def process_role(role_path):
    texts = []

    resume_folder = os.path.join(role_path, "resumes")

    for file in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, file)

        text = extract_text(file_path)
        cleaned = preprocess_text(text)

        texts.append(cleaned)

    # Convert to TF-IDF
    X, vectorizer = create_tfidf_vectors(texts)

    return X, vectorizer


if __name__ == "__main__":
    roles = ["accountant", "chef", "information_technology"]

    for role in roles:
        role_path = f"roles/{role}"

        X, vectorizer = process_role(role_path)

        # Save processed data
        np.save(f"{role_path}/processed/X.npy", X.toarray())
        
        with open(f"{role_path}/processed/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        print(f"{role} processed. Shape:", X.shape)