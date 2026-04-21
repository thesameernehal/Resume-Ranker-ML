from flask import Flask, render_template, request, send_from_directory
from utils.ranker import rank_resumes
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
        top_n = int(request.form.get("top_n", 5))   # ✅ NEW

        if selected_role and jd_text:
            raw_results = rank_resumes(jd_text, selected_role, top_n=top_n)

            # Convert dict → tuple for HTML
            results = [(item["resume"], item["score"]) for item in raw_results]

    return render_template(
        "index.html",
        results=results,
        role=selected_role,
        jd=jd_text
    )


#  View Resume Route
@app.route('/view/<filename>')
def view_file(filename):
    role = request.args.get("role")

    base_dir = os.path.dirname(os.path.abspath(__file__))  # app folder
    project_root = os.path.abspath(os.path.join(base_dir, ".."))

    folder = os.path.join(project_root, "roles", role, "resumes")

    return send_from_directory(folder, filename)

if __name__ == "__main__":
    app.run(debug=True)