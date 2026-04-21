import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessor import preprocess_text
from utils.parser import extract_text_from_file


def rank_uploaded_resumes(file_paths, job_description, top_n=5):

    texts = []
    filenames = []

    # Read all uploaded files
    for path in file_paths:
        text = extract_text_from_file(path)

        if text.strip() == "":
            continue

        texts.append(preprocess_text(text))
        filenames.append(os.path.basename(path))

    # Add JD
    jd_cleaned = preprocess_text(job_description)

    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(texts + [jd_cleaned])

    jd_vector = X[-1]
    resume_vectors = X[:-1]

    similarities = cosine_similarity(jd_vector, resume_vectors)[0]

    results = []
    for file, score in zip(filenames, similarities):
        results.append((file, float(score)))

    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results[:top_n]