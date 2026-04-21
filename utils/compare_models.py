import os
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.prepare_dataset import load_resumes_for_role
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4)
    }


def compare_models_for_role(role):

    texts, labels = load_resumes_for_role(role)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_dir = f"roles/{role}/models"

    models = {
        "Naive Bayes": "nb.pkl",
        "Logistic Regression": "lr.pkl",
        "Random Forest": "rf.pkl"
    }

    role_results = {}

    for name, file in models.items():
        path = os.path.join(model_dir, file)

        if not os.path.exists(path):
            role_results[name] = "Model not found"
            continue

        with open(path, "rb") as f:
            model = pickle.load(f)

        results = evaluate_model(model, X_test, y_test)

        role_results[name] = results

    return role_results


# full comparsion
def get_full_comparison():

    roles = ["accountant", "chef", "information_technology"]

    final_results = {}

    for role in roles:
        final_results[role] = compare_models_for_role(role)

    return final_results