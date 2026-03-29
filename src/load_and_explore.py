import pandas as pd

# Step 1: Load dataset
file_path = "data/raw/UpdatedResumeDataSet.csv"
df = pd.read_csv(file_path)

# Step 2: Basic info
print("Dataset Loaded Successfully!\n")

# Step 3: Show first 5 rows
print("First 5 rows:\n")
print(df.head())

# Step 4: Show columns
print("\nColumns in dataset:")
print(df.columns)

# Step 5: Dataset shape
print("\nDataset shape (rows, columns):")
print(df.shape)

# Step 6: Dataset info
print("\nDataset Info:")
print(df.info())

# Step 7: Category distribution
print("\nCategory counts:")
print(df['Category'].value_counts())