import os

# Limit CPU threads BEFORE importing PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import uuid

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash
)
from werkzeug.utils import secure_filename

from src.predict import predictor

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY", "dermascan-development-key"
)

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        flash("No file selected.")
        return redirect(url_for("index"))

    file = request.files["file"]

    if not file.filename:
        flash("Please select an image.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Only PNG, JPG and JPEG images are allowed.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)

    if not filename:
        flash("Invalid filename.")
        return redirect(url_for("index"))

    # Unique filename prevents upload collisions
    extension = filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{extension}"

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"], filename
    )

    try:
        file.save(filepath)

        # Run AI prediction
        prediction_result, confidence_score = (
            predictor.predict_image(filepath)
        )

        return render_template(
            "result.html",
            prediction=prediction_result,
            confidence=confidence_score,
            image_filename=filename
        )

    except Exception:
        app.logger.exception("Prediction failed")
        flash("Prediction failed. Please try another image.")
        return redirect(url_for("index"))

    finally:
        gc.collect()


@app.errorhandler(413)
def file_too_large(error):
    flash("Image size must be below 5 MB.")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=False)