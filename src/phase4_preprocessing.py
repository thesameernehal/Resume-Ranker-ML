import pandas as pd 


# Loading dataset
dataset = pd.read_csv("data/raw/UpdatedResumeDataSet.csv")

# View first rows 
print(dataset.head()) 

# selecting input roles 
input_roles = ["Data Science" , "HR"]


# Create binary labels
dataset['Label'] = dataset['Category'].apply(lambda x : 1 if x in input_roles else 0)

print(dataset[['Category' , 'Label']].head())


# Text cleaning
import re 
def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)       # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)       # remove emails
    text = re.sub(r'[^a-zA-Z]', ' ', text)     # keep only letters
    text = text.lower()                        # convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()   # remove extra spaces
    return text


dataset['Cleaned_Resume'] = dataset['Resume'].apply(clean_text)

print(dataset[['Resume' , 'Cleaned_Resume']].head())

# Checking label distribution
print(dataset['Label'].value_counts())

# Saving processed data 
dataset.to_csv("data/processed/processed_resume.csv" , index=False)
         
            
            