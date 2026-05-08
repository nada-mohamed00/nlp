from flask import Flask, render_template, request
from nlp_pipeline import rank_candidates

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    job_description = ""
    error = ""
    if request.method == "POST":
        job_description = request.form.get("job_description", "").strip()
        files = request.files.getlist("resumes")
        resumes = [(f.filename, f) for f in files if f and f.filename]
        if not job_description:
            error = "Please enter a job description."
        elif not resumes:
            error = "Please upload at least one resume."
        else:
            results = rank_candidates(job_description, resumes)
    return render_template(
        "index.html",
        results=results,
        job_description=job_description,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
