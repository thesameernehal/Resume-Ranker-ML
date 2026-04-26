from flask import Flask, render_template, request, send_file
from utils.ranker import rank_resumes
from utils.compare_models import get_full_comparison
from utils.rank_uploaded import rank_uploaded_resumes
from utils.ml_ranker import rank_resumes_ml, rank_uploaded_resumes_ml

import os
import shutil

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    results = []
    selected_role = ""
    jd_text = ""
    source_type = None
    model_type = "tfidf"   # default

    if request.method == "POST":

        selected_role = request.form.get("role")
        jd_text = request.form.get("job_description")
        uploaded_files = request.files.getlist("resumes")
        model_type = request.form.get("model_type", "tfidf")

        # safe top_n
        try:
            top_n = int(request.form.get("top_n") or 5)
        except:
            top_n = 5

        print("ROLE:", selected_role)
        print("MODEL:", model_type)

        uploaded_files = [f for f in uploaded_files if f and f.filename.strip() != ""]

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        if jd_text:

            # ================= UPLOADED =================
            if len(uploaded_files) > 0:
                source_type = "upload"

                upload_folder = os.path.join(base_dir, "data")

                if os.path.exists(upload_folder):
                    shutil.rmtree(upload_folder)

                os.makedirs(upload_folder, exist_ok=True)

                saved_paths = []

                for file in uploaded_files:
                    filepath = os.path.join(upload_folder, file.filename)
                    file.save(filepath)
                    saved_paths.append(filepath)

                # MODEL SWITCH
                if model_type == "tfidf":
                    results = rank_uploaded_resumes(saved_paths, jd_text, top_n=top_n)
                else:
                    results = rank_uploaded_resumes_ml(
                        saved_paths, jd_text, selected_role, model_type, top_n
                    )

            # ================= DATASET =================
            elif selected_role:
                source_type = "dataset"

                if model_type == "tfidf":
                    raw_results = rank_resumes(jd_text, selected_role, top_n=top_n)
                    results = [(item["resume"], item["score"]) for item in raw_results]

                else:
                    results = rank_resumes_ml(
                        jd_text, selected_role, model_type, top_n
                    )

    return render_template(
        "index.html",
        results=results,
        role=selected_role,
        jd=jd_text,
        source_type=source_type,
        model_type=model_type
    )


# ================= VIEW =================
@app.route("/view/<filename>")
def view_file(filename):

    role = request.args.get("role")
    source = request.args.get("source")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if source == "upload":
        path = os.path.join(base_dir, "data", filename)
    else:
        path = os.path.join(base_dir, "roles", role, "resumes", filename)

    if not os.path.exists(path):
        return "File not found", 404

    return send_file(path)


# ================= REPORT =================
@app.route("/report")
def report():
    data = get_full_comparison()
    return render_template("report.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)