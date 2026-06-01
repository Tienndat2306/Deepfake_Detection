"""HTTP routes for the deepfake detector web app."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

bp = Blueprint("web", __name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mp4v"}
_service = None


def _get_service():
    global _service
    if _service is None:
        from .inference import DeepfakeInferenceService

        _service = DeepfakeInferenceService(root_dir=ROOT_DIR)
    return _service


def _default_context(active_page: str = "dashboard") -> dict:
    return {
        "active_page": active_page,
        "session_id": "ready",
        "analysis_status": "READY",
        "engine_status": "IDLE",
        "current_frame": 0,
        "video_url": "",
        "video_thumbnail": "",
        "keyframes": [],
    }


@bp.route("/")
@bp.route("/dashboard")
def dashboard():
    return render_template("index.html", **_default_context("dashboard"))


@bp.route("/forensics")
def forensics():
    return render_template("index.html", **_default_context("forensics"))


@bp.route("/deepscan")
def deepscan():
    return render_template("index.html", **_default_context("deepscan"))


@bp.route("/network")
def network():
    return render_template("index.html", **_default_context("network"))


@bp.route("/archive")
def archive():
    return render_template("index.html", **_default_context("archive"))


@bp.route("/api/health")
def health():
    service = _get_service()
    return jsonify(
        {
            "ok": True,
            "device": str(service.device),
            "config": str(service.config_path),
            "model_loaded": service.model is not None,
        }
    )


@bp.route("/api/analyze", methods=["POST"])
def analyze():
    uploaded = request.files.get("video")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"ok": False, "error": "Please select a video file."}), 400

    original_name = secure_filename(uploaded.filename)
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify({"ok": False, "error": f"File format {suffix} is not supported."}), 400

    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    stored_name = f"{session_id}_{original_name}"
    video_path = upload_dir / stored_name
    session_dir = upload_dir / session_id
    uploaded.save(video_path)

    try:
        result = _get_service().analyze_video(video_path=video_path, output_dir=session_dir)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    result["video_url"] = f"/static/uploads/{stored_name}"
    with (session_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return jsonify({"ok": True, "result": result})


@bp.route("/api/session/<session_id>")
def session_result(session_id: str):
    safe_session = secure_filename(session_id)
    result_path = Path(current_app.config["UPLOAD_FOLDER"]) / safe_session / "result.json"
    if not result_path.exists():
        return jsonify({"ok": False, "error": "Khong tim thay session."}), 404
    with result_path.open("r", encoding="utf-8") as f:
        return jsonify({"ok": True, "result": json.load(f)})
