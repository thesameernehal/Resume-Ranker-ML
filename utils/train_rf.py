import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.prepare_dataset import load_resumes_for_role


def train_rf_for_role(role):

    print(f"\n==============================")
    print(f"Training Random Forest for: {role}")
    print(f"==============================\n")

    # Load dataset
    texts, labels = load_resumes_for_role(role)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    y = labels

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save inside role folder
    model_dir = f"roles/{role}/models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "rf.pkl")
    vectorizer_path = os.path.join(model_dir, "tfidf_rf.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nSaved model at: {model_path}")
    print(f"Saved vectorizer at: {vectorizer_path}\n")


def train_all_rf_models():
    roles = ["accountant", "chef", "information_technology"]

    for role in roles:
        train_rf_for_role(role)


if __name__ == "__main__":
    train_all_rf_models()