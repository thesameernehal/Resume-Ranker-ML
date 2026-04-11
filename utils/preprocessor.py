import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Remove special characters & numbers
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Tokenization (split words)
    words = text.split()

    # Remove stopwords
    filtered_words = [word for word in words if word not in stop_words]

    # Join back to text
    cleaned_text = " ".join(filtered_words)

    return cleaned_text