import os
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.prepare_dataset import load_resumes_for_role
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


def compare_models_for_role(role):

    # Load dataset
    texts, labels = load_resumes_for_role(role)

    # Use same TF-IDF for fair comparison
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    y = labels

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load models
    model_dir = f"roles/{role}/models"

    models = {
        "Naive Bayes": "nb.pkl",
        "Logistic Regression": "lr.pkl",
        "Random Forest": "rf.pkl"
    }

    for name, file in models.items():
        path = os.path.join(model_dir, file)

        if not os.path.exists(path):
            print(f"{name}: Model not found!")
            continue

        with open(path, "rb") as f:
            model = pickle.load(f)

        results = evaluate_model(model, X_test, y_test)

        print(f"\n{name}:")
        print(f"Accuracy : {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall   : {results['recall']:.4f}")
        print(f"F1-score : {results['f1']:.4f}")


def run_comparison():
    roles = ["accountant", "chef", "information_technology"]

    for role in roles:
     print("\n" + "="*30)
     print(f"{role.upper()}")
     print("="*30)

     compare_models_for_role(role)


if __name__ == "__main__":
    run_comparison()