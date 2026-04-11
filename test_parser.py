from utils.parser import extract_text

# Testing all formats 
file_path = "roles/accountant/resumes/accountant_resume_21.pdf"
# file_path = "roles/accountant/resumes/accountant_resume_1.txt"
# file_path = "roles/accountant/resumes/accountant_resume_17.docx"

text = extract_text(file_path)
print(text[:500])  # print first 500 characters