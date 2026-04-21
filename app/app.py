from flask import Flask, render_template, request, send_file
from utils.ranker import rank_resumes
from utils.compare_models import get_full_comparison
from utils.rank_uploaded import rank_uploaded_resumes
import os

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    results = []
    selected_role = ""
    jd_text = ""

    if request.method == "POST":
        selected_role = request.form.get("role")
        jd_text = request.form.get("job_description")
        uploaded_files = request.files.getlist("resumes")

        # Filter valid files
        uploaded_files = [f for f in uploaded_files if f and f.filename.strip() != ""]

        if jd_text:

            # CASE 1: Uploaded resumes
            if len(uploaded_files) > 0:
                print("Using uploaded resumes...")

                os.makedirs("data", exist_ok=True)
                saved_paths = []

                for file in uploaded_files:
                    path = os.path.join("data", file.filename)
                    file.save(path)
                    saved_paths.append(path)

                raw_results = rank_uploaded_resumes(saved_paths, jd_text, top_n=5)

                # Convert to (file, score, source)
                results = [
                    (os.path.basename(path), score, "upload")
                    for path, score in raw_results
                ]

            # CASE 2: Dataset resumes
            elif selected_role:
                print("Using dataset resumes...")

                raw_results = rank_resumes(jd_text, selected_role, top_n=5)

                results = [
                    (item["resume"], item["score"], "dataset")
                    for item in raw_results
                ]

    return render_template(
        "index.html",
        results=results,
        role=selected_role,
        jd=jd_text
    )


# View Resume Route
@app.route("/view/<filename>")
def view_file(filename):
    role = request.args.get("role")
    source = request.args.get("source")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if source == "upload":
        path = os.path.join(base_dir, "data", filename)
    else:
        path = os.path.join(base_dir, "roles", role, "resumes", filename)

    return send_file(path)

# Report Route
@app.route("/report")
def report():
    data = get_full_comparison()
    return render_template("report.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)