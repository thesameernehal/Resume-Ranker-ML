import os
import pickle
from utils.parser import extract_text


def load_model_and_vectorizer(role, model_type):
    base_path = os.path.join("roles", role, "models")

    if model_type == "nb":
        model_path = os.path.join(base_path, "nb.pkl")
        vec_path = os.path.join(base_path, "tfidf.pkl")

    elif model_type == "lr":
        model_path = os.path.join(base_path, "lr.pkl")
        vec_path = os.path.join(base_path, "tfidf_lr.pkl")

    elif model_type == "rf":
        model_path = os.path.join(base_path, "rf.pkl")
        vec_path = os.path.join(base_path, "tfidf_rf.pkl")

    else:
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# For dataset resumes
def rank_resumes_ml(role, model_type, resumes_folder, top_n=5):

    model, vectorizer = load_model_and_vectorizer(role, model_type)

    if model is None:
        return []

    results = []

    for file in os.listdir(resumes_folder):
        path = os.path.join(resumes_folder, file)

        try:
            text = extract_text(path)
            X = vectorizer.transform([text])

            prob = model.predict_proba(X)[0]
            score = prob[1] if len(prob) > 1 else prob[0]

            results.append((file, score))

        except Exception as e:
            print(f"Error processing {file}: {e}")

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


# For uploaded resumes
def rank_uploaded_resumes_ml(file_paths, role, model_type, top_n=5):

    model, vectorizer = load_model_and_vectorizer(role, model_type)

    if model is None:
        return []

    results = []

    for path in file_paths:
        try:
            filename = os.path.basename(path)

            text = extract_text(path)
            X = vectorizer.transform([text])

            prob = model.predict_proba(X)[0]
            score = prob[1] if len(prob) > 1 else prob[0]

            results.append((filename, score))

        except Exception as e:
            print(f"Error processing {path}: {e}")

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]