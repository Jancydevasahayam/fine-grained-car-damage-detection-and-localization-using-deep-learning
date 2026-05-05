from __future__ import annotations

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from utils.csv_utils import (
    REPORT_FIELDS,
    append_row,
    get_average_part_price,
    get_brand_model_catalog,
    get_part_price,
    initialize_data_files,
    read_rows,
)
from utils.model_utils import (
    infer_severity,
    load_models as load_cnn_models,
    predict_car_presence as infer_car_presence,
    predict_car_presence_from_rgb,
    predict_damage as infer_damage,
    predict_damage_from_rgb,
    predict_part_with_details,
    predict_part_with_details_from_rgb,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
SEVERITY_MULTIPLIERS = {
    "Less": 0.75,
    "Moderate": 1.0,
    "Severe": 1.35,
}
PART_DISPLAY_NAMES = {
    "bonnet": "Bonnet",
    "bumper": "Bumper",
    "door": "Door",
    "fender": "Fender",
    "light": "Light",
    "windshield": "Windshield",
    "unknown": "Unknown",
}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "car-damage-detection-secret")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILES = initialize_data_files(DATA_DIR)


def load_models():
    """Load TensorFlow/Keras models for car presence, damage, and part detection."""
    return load_cnn_models(MODELS_DIR)


MODELS = load_models()
MODEL_FALLBACK_ACTIVE = any(model is None for model in MODELS.values())


def predict_car_presence(image_path: str):
    return infer_car_presence(MODELS.get("car_presence"), image_path)


def predict_damage(image_path: str):
    return infer_damage(MODELS.get("damage"), image_path)


def to_display_part(part: str) -> str:
    return PART_DISPLAY_NAMES.get((part or "").strip().lower(), (part or "").title())


def calculate_cost(
    brand: str, model: str, part: str, severity: str, damage_status: str
) -> tuple[float, str]:
    """Calculate cost from CSV price table and severity multiplier."""
    if (damage_status or "").strip().lower() != "damaged":
        return 0.0, "not_damaged"

    if (part or "").strip().lower() == "unknown":
        return 0.0, "part_uncertain"

    base_price = get_part_price(CSV_FILES["prices"], brand, model, part)
    price_source = "exact"

    if base_price is None:
        base_price = get_average_part_price(CSV_FILES["prices"], part)
        price_source = "average"

    if base_price is None:
        return 0.0, "unavailable"

    multiplier = SEVERITY_MULTIPLIERS.get(severity, 1.0)
    return round(base_price * multiplier, 2), price_source


def save_to_csv(report_row: dict):
    append_row(CSV_FILES["reports"], REPORT_FIELDS, report_row)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _render_dashboard(
    uploaded_image: str | None = None,
    owner_name: str = "",
    selected_brand: str = "",
    selected_model: str = "",
):
    return render_template(
        "dashboard.html",
        uploaded_image=uploaded_image,
        owner_name=owner_name,
        selected_brand=selected_brand,
        selected_model=selected_model,
        brand_model_catalog=get_brand_model_catalog(CSV_FILES["prices"]),
        model_fallback_active=MODEL_FALLBACK_ACTIVE,
    )


def _as_percent(value: float) -> str:
    return f"{value * 100:.2f}"


def _encode_status(
    damage_status: str,
    severity: str,
    brand: str,
    model: str,
    price_source: str,
    part_confidence: float,
    part_uncertain: bool,
) -> str:
    return (
        f"{damage_status}|{severity}|{brand}|{model}|{price_source}|"
        f"{part_confidence:.4f}|{int(part_uncertain)}"
    )


def _parse_report_rows() -> list[Dict[str, str]]:
    reports = []
    for row in reversed(read_rows(CSV_FILES["reports"])):
        status_raw = (row.get("damage_status") or "").split("|")
        damage_status = status_raw[0].title() if len(status_raw) > 0 and status_raw[0] else "-"
        severity = status_raw[1] if len(status_raw) > 1 and status_raw[1] else "-"
        brand = status_raw[2] if len(status_raw) > 2 and status_raw[2] else "-"
        model = status_raw[3] if len(status_raw) > 3 and status_raw[3] else "-"
        price_source = status_raw[4] if len(status_raw) > 4 and status_raw[4] else "-"

        part_conf = "-"
        if len(status_raw) > 5 and status_raw[5]:
            try:
                part_conf = f"{float(status_raw[5]) * 100:.2f}%"
            except ValueError:
                part_conf = "-"

        uncertain_text = "-"
        if len(status_raw) > 6 and status_raw[6]:
            uncertain_text = "Yes" if status_raw[6] == "1" else "No"

        reports.append(
            {
                "owner_name": row.get("email", "Guest User"),
                "image_path": row.get("image_path", ""),
                "detected_part": to_display_part(row.get("detected_part", "")),
                "damage_status": damage_status,
                "severity": severity,
                "brand": brand,
                "model": model,
                "price_source": price_source,
                "part_confidence": part_conf,
                "part_uncertain": uncertain_text,
                "estimated_cost": row.get("estimated_cost", "0.00"),
                "timestamp": row.get("timestamp", ""),
            }
        )

    return reports


def _render_login_removed_redirect():
    flash("User login has been removed. Please continue from the dashboard.", "info")
    return redirect(url_for("dashboard"))


def _decode_data_url_image(data_url: str) -> Tuple[np.ndarray, np.ndarray]:
    payload = (data_url or "").strip()
    if not payload:
        raise ValueError("Camera frame is empty.")

    encoded = payload.split(",", 1)[1] if "," in payload else payload
    try:
        image_bytes = base64.b64decode(encoded)
    except Exception as error:
        raise ValueError("Unable to decode camera frame.") from error

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Unable to read camera frame.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, image_bgr


def _save_rgb_image(image_bgr: np.ndarray, filename_prefix: str) -> Tuple[str, str]:
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{filename_prefix}.jpg"
    file_path = UPLOAD_DIR / filename
    cv2.imwrite(str(file_path), image_bgr)
    return str(file_path), f"uploads/{filename}"


def _top_part_suggestions(top_parts: List[Tuple[str, float]]) -> List[Dict[str, str]]:
    suggestions = []
    for label, confidence in top_parts:
        suggestions.append(
            {
                "part": to_display_part(label),
                "confidence": f"{confidence * 100:.2f}",
            }
        )
    return suggestions


def _build_assessment_from_path(
    image_path: str,
    relative_image_path: str,
    source_name: str,
    owner_name: str,
    car_brand: str,
    car_model: str,
) -> Dict[str, object]:
    car_status, car_confidence = predict_car_presence(image_path)
    if car_status != "car":
        return {
            "source_name": source_name,
            "image_path": relative_image_path,
            "status": "no_car",
            "detected_part": "N/A",
            "damage_status": "No Car",
            "severity": "N/A",
            "estimated_cost": "0.00",
            "price_source": "no_car",
            "car_confidence": _as_percent(car_confidence),
            "damage_confidence": "0.00",
            "part_confidence": "0.00",
            "part_uncertain": True,
            "part_top3": [],
            "note": "No car detected in this image.",
            "car_brand": car_brand,
            "car_model": car_model,
        }

    damage_status, damage_confidence = predict_damage(image_path)
    severity = infer_severity(damage_status, damage_confidence)
    part_label, part_confidence, top_parts, uncertain = predict_part_with_details(
        MODELS.get("part"), image_path
    )
    estimated_cost, price_source = calculate_cost(
        car_brand, car_model, part_label, severity, damage_status
    )

    note = ""
    if uncertain:
        note = "Part prediction confidence is low. Check Top 3 suggestions before finalizing."

    save_to_csv(
        {
            "email": owner_name,
            "image_path": relative_image_path,
            "detected_part": part_label,
            "damage_status": _encode_status(
                damage_status,
                severity,
                car_brand,
                car_model,
                price_source,
                part_confidence,
                uncertain,
            ),
            "estimated_cost": f"{estimated_cost:.2f}",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )

    return {
        "source_name": source_name,
        "image_path": relative_image_path,
        "status": "ok",
        "detected_part": to_display_part(part_label),
        "damage_status": damage_status.title(),
        "severity": severity,
        "estimated_cost": f"{estimated_cost:.2f}",
        "price_source": price_source,
        "car_confidence": _as_percent(car_confidence),
        "damage_confidence": _as_percent(damage_confidence),
        "part_confidence": _as_percent(part_confidence),
        "part_uncertain": uncertain,
        "part_top3": _top_part_suggestions(top_parts),
        "note": note,
        "car_brand": car_brand,
        "car_model": car_model,
    }


def _build_assessment_from_rgb(
    image_rgb: np.ndarray,
    image_relative_path: str,
    source_name: str,
    owner_name: str,
    car_brand: str,
    car_model: str,
    save_report: bool,
) -> Dict[str, object]:
    car_status, car_confidence = predict_car_presence_from_rgb(
        MODELS.get("car_presence"), image_rgb
    )
    if car_status != "car":
        return {
            "source_name": source_name,
            "image_path": image_relative_path,
            "status": "no_car",
            "detected_part": "N/A",
            "damage_status": "No Car",
            "severity": "N/A",
            "estimated_cost": "0.00",
            "price_source": "no_car",
            "car_confidence": _as_percent(car_confidence),
            "damage_confidence": "0.00",
            "part_confidence": "0.00",
            "part_uncertain": True,
            "part_top3": [],
            "note": "No car detected in this frame.",
            "car_brand": car_brand,
            "car_model": car_model,
        }

    damage_status, damage_confidence = predict_damage_from_rgb(MODELS.get("damage"), image_rgb)
    severity = infer_severity(damage_status, damage_confidence)
    part_label, part_confidence, top_parts, uncertain = predict_part_with_details_from_rgb(
        MODELS.get("part"), image_rgb
    )
    estimated_cost, price_source = calculate_cost(
        car_brand, car_model, part_label, severity, damage_status
    )

    if save_report:
        save_to_csv(
            {
                "email": owner_name,
                "image_path": image_relative_path,
                "detected_part": part_label,
                "damage_status": _encode_status(
                    damage_status,
                    severity,
                    car_brand,
                    car_model,
                    price_source,
                    part_confidence,
                    uncertain,
                ),
                "estimated_cost": f"{estimated_cost:.2f}",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

    note = ""
    if uncertain:
        note = "Part prediction confidence is low. Check Top 3 suggestions before finalizing."

    return {
        "source_name": source_name,
        "image_path": image_relative_path,
        "status": "ok",
        "detected_part": to_display_part(part_label),
        "damage_status": damage_status.title(),
        "severity": severity,
        "estimated_cost": f"{estimated_cost:.2f}",
        "price_source": price_source,
        "car_confidence": _as_percent(car_confidence),
        "damage_confidence": _as_percent(damage_confidence),
        "part_confidence": _as_percent(part_confidence),
        "part_uncertain": uncertain,
        "part_top3": _top_part_suggestions(top_parts),
        "note": note,
        "car_brand": car_brand,
        "car_model": car_model,
    }


def _collect_image_sources() -> List[Dict[str, object]]:
    sources: List[Dict[str, object]] = []
    files = [f for f in request.files.getlist("images") if f and f.filename]
    if not files:
        single_file = request.files.get("image")
        if single_file and single_file.filename:
            files = [single_file]

    for image_file in files:
        sources.append({"type": "file", "file": image_file})

    camera_image = (request.form.get("cameraImage") or "").strip()
    if camera_image:
        sources.append({"type": "camera", "data": camera_image})

    return sources


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if request.method == "POST":
        owner_name = (request.form.get("ownerName") or "").strip() or "Guest User"
        car_brand = (request.form.get("carBrand") or "").strip()
        car_model = (request.form.get("carModel") or "").strip()
        brand_model_catalog = get_brand_model_catalog(CSV_FILES["prices"])
        sources = _collect_image_sources()

        if not car_brand or not car_model:
            flash("Please select both car brand and model.", "error")
            return _render_dashboard(owner_name=owner_name, selected_brand=car_brand, selected_model=car_model)

        if car_brand not in brand_model_catalog:
            flash("Selected brand is not available in the price catalog.", "error")
            return _render_dashboard(owner_name=owner_name, selected_brand=car_brand, selected_model=car_model)

        if car_model not in brand_model_catalog[car_brand]:
            flash("Selected model does not match the selected brand.", "error")
            return _render_dashboard(owner_name=owner_name, selected_brand=car_brand, selected_model=car_model)

        if not sources:
            flash("Please upload at least one image or capture a webcam frame.", "error")
            return _render_dashboard(owner_name=owner_name, selected_brand=car_brand, selected_model=car_model)

        if len(sources) > 15:
            flash("Please upload up to 15 images at a time.", "error")
            return _render_dashboard(owner_name=owner_name, selected_brand=car_brand, selected_model=car_model)

        assessments: List[Dict[str, object]] = []
        invalid_files = 0

        for index, source in enumerate(sources, start=1):
            source_type = source["type"]
            try:
                if source_type == "file":
                    image_file = source["file"]
                    if not isinstance(image_file, FileStorage):
                        continue
                    if not allowed_file(image_file.filename):
                        invalid_files += 1
                        continue

                    filename = (
                        f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_"
                        f"{secure_filename(image_file.filename)}"
                    )
                    image_path = UPLOAD_DIR / filename
                    image_file.save(image_path)
                    relative_image_path = f"uploads/{filename}"

                    assessment = _build_assessment_from_path(
                        str(image_path),
                        relative_image_path,
                        source_name=image_file.filename or f"image_{index}",
                        owner_name=owner_name,
                        car_brand=car_brand,
                        car_model=car_model,
                    )
                    assessments.append(assessment)

                else:
                    image_rgb, image_bgr = _decode_data_url_image(str(source.get("data", "")))
                    _, relative_image_path = _save_rgb_image(image_bgr, "webcam")
                    assessment = _build_assessment_from_rgb(
                        image_rgb=image_rgb,
                        image_relative_path=relative_image_path,
                        source_name=f"webcam_frame_{index}",
                        owner_name=owner_name,
                        car_brand=car_brand,
                        car_model=car_model,
                        save_report=True,
                    )
                    assessments.append(assessment)

            except ValueError as error:
                assessments.append(
                    {
                        "source_name": f"source_{index}",
                        "image_path": "",
                        "status": "error",
                        "detected_part": "N/A",
                        "damage_status": "Error",
                        "severity": "N/A",
                        "estimated_cost": "0.00",
                        "price_source": "error",
                        "car_confidence": "0.00",
                        "damage_confidence": "0.00",
                        "part_confidence": "0.00",
                        "part_uncertain": True,
                        "part_top3": [],
                        "note": str(error),
                        "car_brand": car_brand,
                        "car_model": car_model,
                    }
                )

        if invalid_files > 0:
            flash(f"Skipped {invalid_files} invalid file(s). Use PNG/JPG/JPEG only.", "info")

        if not assessments:
            flash("No valid images could be processed.", "error")
            return _render_dashboard(owner_name=owner_name, selected_brand=car_brand, selected_model=car_model)

        if MODEL_FALLBACK_ACTIVE:
            flash(
                "One or more model files are missing. Predictions may be less accurate until models are added.",
                "info",
            )

        uncertain_count = sum(1 for item in assessments if item.get("part_uncertain"))
        no_car_count = sum(1 for item in assessments if item.get("status") == "no_car")

        return render_template(
            "estimate.html",
            owner_name=owner_name,
            car_brand=car_brand,
            car_model=car_model,
            assessments=assessments,
            total_count=len(assessments),
            uncertain_count=uncertain_count,
            no_car_count=no_car_count,
            model_fallback_active=MODEL_FALLBACK_ACTIVE,
        )

    return _render_dashboard()


@app.route("/api/live-detect", methods=["POST"])
def live_detect():
    payload = request.get_json(silent=True) or {}
    owner_name = (payload.get("ownerName") or "Live User").strip()
    car_brand = (payload.get("carBrand") or "").strip()
    car_model = (payload.get("carModel") or "").strip()
    data_url = (payload.get("image") or "").strip()
    save_snapshot = str(payload.get("saveSnapshot", "false")).strip().lower() in {
        "1",
        "true",
        "yes",
    }
    brand_model_catalog = get_brand_model_catalog(CSV_FILES["prices"])

    if not car_brand or not car_model:
        return jsonify({"ok": False, "error": "Please provide car brand and model."}), 400

    if car_brand not in brand_model_catalog or car_model not in brand_model_catalog.get(car_brand, []):
        return jsonify({"ok": False, "error": "Brand/model mismatch in catalog."}), 400

    if not data_url:
        return jsonify({"ok": False, "error": "No camera image provided."}), 400

    try:
        image_rgb, image_bgr = _decode_data_url_image(data_url)
    except ValueError as error:
        return jsonify({"ok": False, "error": str(error)}), 400

    image_rel_path = ""
    if save_snapshot:
        _, image_rel_path = _save_rgb_image(image_bgr, "live")

    assessment = _build_assessment_from_rgb(
        image_rgb=image_rgb,
        image_relative_path=image_rel_path,
        source_name="live_webcam",
        owner_name=owner_name,
        car_brand=car_brand,
        car_model=car_model,
        save_report=save_snapshot,
    )

    return jsonify({"ok": True, "result": assessment})


@app.route("/results")
def results():
    return render_template("results.html", reports=_parse_report_rows()[:20])


@app.route("/login", methods=["GET", "POST"])
def login():
    return _render_login_removed_redirect()


@app.route("/signup", methods=["GET", "POST"])
def signup():
    return _render_login_removed_redirect()


@app.route("/logout")
def logout():
    return _render_login_removed_redirect()


@app.route("/view_profile")
def view_profile():
    return _render_login_removed_redirect()


@app.route("/edit_profile", methods=["GET", "POST"])
def edit_profile():
    return _render_login_removed_redirect()


@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)
