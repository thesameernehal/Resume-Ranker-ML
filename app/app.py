from flask import Flask, render_template, request
from utils.ranker import rank_resumes

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    results = []
    selected_role = ""
    jd_text = ""

    if request.method == "POST":
        selected_role = request.form.get("role")
        jd_text = request.form.get("job_description")

        if selected_role and jd_text:
           results = rank_resumes(jd_text, selected_role, top_n=5)

    return render_template(
        "index.html",
        results=results,
        role=selected_role,
        jd=jd_text
    )


if __name__ == "__main__":
    app.run(debug=True)