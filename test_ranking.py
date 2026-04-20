from utils.ranker import rank_resumes

# Sample Job Description
job_description = """
The Accountant is responsible for maintaining accurate financial records, managing tax compliance, and preparing comprehensive financial reports to ensure the organization's fiscal health. Key duties include reconciling bank statements, managing accounts payable and receivable, preparing monthly and annual balance sheets, and overseeing payroll processing. The role requires a strong understanding of Generally Accepted Accounting Principles (GAAP) to conduct internal audits and ensure all transactions align with regulatory standards. Additionally, the Accountant assists in budget forecasting, tracks organizational spending, and provides data-driven financial insights to senior management. The ideal candidate possesses a bachelor’s degree in Accounting or Finance, proficiency in software like QuickBooks or SAP, and advanced Excel skills such as pivot tables and VLOOKUPs to streamline financial operations.
"""

# Role selection
role = "accountant"

# Run ranking
results = rank_resumes(job_description, role, top_n=5, threshold=0.05)

# Print results
print("\nTop Ranked Resumes:\n")

for i, (resume, score) in enumerate(results, start=1):
    print(f"{i}. {resume} → Score: {score:.4f}")