import pandas as pd

# Step 1 - Loading Processed Dataset
dataset = pd.read_csv("data\processed\processed_resume.csv")
print(dataset.head())

# Step 2 - Selecting features and labels
X = dataset['Cleaned_Resume']
y = dataset['Label']
print(X.head())
print(y.head())

# Step 3 - applying TF-IDF Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer(max_features=5000)
X_vectorized = tf_idf.fit_transform(X)
print(X_vectorized.shape)

# Step 4 - Convert to array
X_array = X_vectorized.toarray()

# Step 5 - Saving Vectorizer
import pickle
with open ("models/tfidf_vectorizer.pkl","wb") as f:
    pickle.dump(tf_idf,f)
    
# Step 6 - Save features and labels
import numpy as np
np.save("data/processed/X.npy" , X_array)
np.save("data/processed/y.npy" , y)
