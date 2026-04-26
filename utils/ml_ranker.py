import os
import pickle

from utils.parser import extract_text


def load_model_and_vectorizer(role, model_type):
    model_dir = os.path.join("roles", role, "models")

    if model_type == "lr":
        model_path = os.path.join(model_dir, "lr.pkl")
        vectorizer_path = os.path.join(model_dir, "tfidf_lr.pkl")

    elif model_type == "nb":
        model_path = os.path.join(model_dir, "nb.pkl")
        vectorizer_path = os.path.join(model_dir, "tfidf.pkl")

    elif model_type == "rf":
        model_path = os.path.join(model_dir, "rf.pkl")
        vectorizer_path = os.path.join(model_dir, "tfidf_rf.pkl")

    else:
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def rank_resumes_ml(file_paths, jd_text, role, model_type="lr", top_n=5):

    model, vectorizer = load_model_and_vectorizer(role, model_type)

    if model is None or vectorizer is None:
        return []

    results = []

    # Transform JD once
    jd_vector = vectorizer.transform([jd_text])

    for path in file_paths:
        try:
            text = extract_text(path)

            if not text:
                continue

            resume_vector = vectorizer.transform([text])

            # Combine JD + resume (simple trick)
            combined = jd_vector + resume_vector

            # Get probability
            prob = model.predict_proba(combined)[0][1]

            filename = os.path.basename(path)

            results.append((filename, prob))

        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Sort by score
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_n]