from utils.ranker import rank_resumes

# Sample Job Description
job_description = """
Looking for a software developer skilled in Python, machine learning,
data analysis, and web development. Experience with Flask is a plus.
"""

# Role selection
role = "chef"

# Run ranking
results = rank_resumes(job_description, role, top_n=5, threshold=0.05)

# Print results
print("\nTop Ranked Resumes:\n")

for i, (resume, score) in enumerate(results, start=1):
    print(f"{i}. {resume} → Score: {score:.4f}")