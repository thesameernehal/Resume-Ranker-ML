from utils.parser import extract_text
from utils.preprocessor import preprocess_text

# Testing all formats 
# file_path = "roles/accountant/resumes/accountant_resume_21.pdf"
# file_path = "roles/accountant/resumes/accountant_resume_1.txt"
file_path = "roles/chef/resumes/chef_resume_41.pdf"

text = extract_text(file_path)
cleaned = preprocess_text(text)

print(cleaned[:500]) # print first 500 characters
