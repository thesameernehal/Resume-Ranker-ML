from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_vectors(text_list):
    vectorizer = TfidfVectorizer(max_features=5000)

    X = vectorizer.fit_transform(text_list)

    return X, vectorizer