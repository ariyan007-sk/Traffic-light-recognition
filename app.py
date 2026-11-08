"""
TRAFFIC AI - Intelligent Traffic Light & Road Sign Recognition System
Flask Web Server & REST API Gateway

Project-Based Learning (PBL) in Artificial Intelligence
"""

import os
import sys
import time
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    PORT, HOST, DEBUG, SECRET_KEY, UPLOAD_DIR,
    SAMPLE_DIR, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH,
    MODEL_NAME
)
from services.detection import detect_traffic_lights, get_yolo_model
from services.history import record_detection, fetch_history, fetch_stats, purge_history
from services.auth_service import send_otp, verify_otp_and_login
from services.sign_classifier import SIGN_CATALOG

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrafficAIApp")

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR)
)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# State tracker for throttling webcam database logging
_last_webcam_log_time = 0.0
_last_webcam_logged_signal = None


def allowed_file(filename: str) -> bool:
    """Verifies image file extension against allowed whitelist."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_base64_image(b64_str: str) -> Optional[Any]:
    """Decodes data URI or raw base64 string to an OpenCV BGR image array."""
    try:
        import cv2
        import numpy as np
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None


@app.context_processor
def inject_user_context():
    """Injects current logged-in user into all Jinja templates."""
    return {
        "current_user": session.get("user", None)
    }


# ==========================================
# PAGE ROUTES
# ==========================================

@app.route("/")
def index():
    """Main Dashboard with real-time statistics, active status, and quick links."""
    stats = fetch_stats()
    recent_history = fetch_history(limit=5)
    return render_template("dashboard.html", stats=stats, recent_history=recent_history, page="dashboard")


@app.route("/dashboard")
def dashboard():
    """Redirect to main dashboard."""
    return redirect(url_for("index"))


@app.route("/image-detection")
def image_detection():
    """Dedicated Image Recognition module with drag-and-drop and sample selection."""
    return render_template("image_detect.html", page="image_detect")


@app.route("/webcam-detection")
def webcam_detection():
    """Real-time live video recognition module using WebRTC / getUserMedia API."""
    return render_template("webcam_detect.html", page="webcam_detect")


@app.route("/history")
def history_page():
    """Detection history audit log page with filtering and clear controls."""
    color_filter = request.args.get("color", "")
    category_filter = request.args.get("category", "")
    history_records = fetch_history(
        limit=100,
        filter_color=color_filter if color_filter else None,
        filter_category=category_filter if category_filter else None
    )
    stats = fetch_stats()
    return render_template(
        "history.html",
        history=history_records,
        stats=stats,
        active_filter=color_filter,
        active_category=category_filter,
        page="history"
    )


@app.route("/about")
def about_page():
    """Comprehensive Project-Based Learning (PBL) report page."""
    return render_template("about.html", page="about")


@app.route("/login")
def login_page():
    """Dedicated Authentication page for Gmail & Mobile OTP sign-in."""
    if "user" in session:
        return redirect(url_for("index"))
    return render_template("login.html", page="login")


# ==========================================
# AUTHENTICATION & OTP API ENDPOINTS
# ==========================================

@app.route("/api/auth/send-otp", methods=["POST"])
def api_send_otp():
    """
    Sends a 6-digit OTP to the user's Gmail address or Mobile number.
    In development mode, returns the code in response for rapid testing.
    """
    data = request.get_json(silent=True) or {}
    identifier = data.get("identifier", "").strip()
    auth_type = data.get("auth_type", "gmail").strip().lower()

    if not identifier:
        return jsonify({"success": False, "error": "Please provide an email address or mobile number."}), 400

    result = send_otp(identifier, auth_type)
    status_code = 200 if result["success"] else 400
    return jsonify(result), status_code


@app.route("/api/auth/verify-otp", methods=["POST"])
def api_verify_otp():
    """
    Verifies user-entered 6-digit OTP code and establishes an authenticated user session.
    """
    data = request.get_json(silent=True) or {}
    identifier = data.get("identifier", "").strip()
    auth_type = data.get("auth_type", "gmail").strip().lower()
    otp_code = data.get("otp", "").strip()

    result = verify_otp_and_login(identifier, auth_type, otp_code)
    if result["success"]:
        # Persist user in session cookie
        session["user"] = result["user"]
        logger.info(f"User authenticated: {result['user']['identifier']} ({result['user']['auth_type']})")

    status_code = 200 if result["success"] else 400
    return jsonify(result), status_code


@app.route("/api/auth/logout", methods=["POST", "GET"])
def api_logout():
    """Terminates active session."""
    session.pop("user", None)
    if request.method == "POST" or request.is_json:
        return jsonify({"success": True, "message": "Signed out successfully."})
    return redirect(url_for("index"))


@app.route("/api/auth/current-user")
def api_current_user():
    """Returns currently authenticated user profile."""
    if "user" in session:
        return jsonify({"logged_in": True, "user": session["user"]})
    return jsonify({"logged_in": False, "user": None})


# ==========================================
# DETECTION REST API ENDPOINTS
# ==========================================

@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    """
    Analyzes an uploaded image for Traffic Lights and Road Signs.
    Accepts multipart file upload ('image') or JSON with 'image' (base64 string).
    Returns bounding box telemetry, classifications, and annotated base64 image.
    """
    import cv2
    import numpy as np

    image_bgr = None

    if "image" in request.files:
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"success": False, "error": "No selected file."}), 400
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": f"Invalid file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    elif request.is_json and "image" in request.json:
        b64_data = request.json["image"]
        image_bgr = decode_base64_image(b64_data)

    if image_bgr is None:
        return jsonify({"success": False, "error": "Failed to decode valid image data."}), 400

    try:
        # Run AI detection for traffic lights and road signs
        result = detect_traffic_lights(image_bgr)

        # Retrieve logged-in user
        user_id = session.get("user", {}).get("identifier", "Guest")

        # Log into SQLite history
        record_detection(
            mode="Image",
            lights_detected=result["total_count"],
            signal_color=result["primary_signal"],
            instruction=result["primary_instruction"],
            yolo_confidence=result["yolo_confidence"],
            color_confidence=result["color_confidence"],
            status="Success" if result["detected"] else "No Traffic Target Detected",
            category=result.get("category", "Traffic Light"),
            sign_name=result.get("sign_name", ""),
            user_identifier=user_id
        )

        return jsonify(result)

    except Exception as e:
        logger.exception("Inference error in api_detect_image:")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/detect/frame", methods=["POST"])
def api_detect_frame():
    """
    Processes a single frame from the live webcam stream.
    Accepts JSON payload: { "frame": "data:image/jpeg;base64,..." }
    """
    global _last_webcam_log_time, _last_webcam_logged_signal

    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"success": False, "error": "Missing 'frame' data in request."}), 400

    frame_bgr = decode_base64_image(data["frame"])
    if frame_bgr is None:
        return jsonify({"success": False, "error": "Could not decode webcam frame."}), 400

    try:
        result = detect_traffic_lights(frame_bgr, conf_threshold=0.35)

        # Smart SQLite throttling
        current_time = time.time()
        should_log = False

        if result["detected"]:
            target_id = f"{result.get('category')}_{result.get('primary_signal')}"
            if target_id != _last_webcam_logged_signal or (current_time - _last_webcam_log_time) > 3.0:
                should_log = True
                _last_webcam_logged_signal = target_id
                _last_webcam_log_time = current_time

        if should_log:
            user_id = session.get("user", {}).get("identifier", "Guest")
            record_detection(
                mode="Webcam",
                lights_detected=result["total_count"],
                signal_color=result["primary_signal"],
                instruction=result["primary_instruction"],
                yolo_confidence=result["yolo_confidence"],
                color_confidence=result["color_confidence"],
                status="Success",
                category=result.get("category", "Traffic Light"),
                sign_name=result.get("sign_name", ""),
                user_identifier=user_id
            )

        return jsonify(result)

    except Exception as e:
        logger.exception("Webcam frame inference error:")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Returns real-time aggregate detection metrics."""
    stats = fetch_stats()
    return jsonify(stats)


@app.route("/api/history")
def api_history():
    """Returns detection history list with optional filtering."""
    color_filter = request.args.get("color", None)
    category_filter = request.args.get("category", None)
    limit = int(request.args.get("limit", 50))
    history = fetch_history(limit=limit, filter_color=color_filter, filter_category=category_filter)
    return jsonify({"success": True, "count": len(history), "history": history})


@app.route("/api/history/clear", methods=["POST"])
def api_clear_history():
    """Clears all logged detection records."""
    deleted_count = purge_history()
    return jsonify({"success": True, "deleted_count": deleted_count, "message": "History cleared successfully."})


@app.route("/api/samples")
def api_samples():
    """Returns a list of bundled sample traffic lights and road signs for one-click testing."""
    samples = []
    if SAMPLE_DIR.exists():
        for file in sorted(SAMPLE_DIR.glob("*.*")):
            if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                clean_name = file.stem.replace("sample_", "").replace("_", " ").title()
                is_sign = "Sign" in clean_name
                category = "Road Sign" if is_sign else "Traffic Light"
                samples.append({
                    "filename": file.name,
                    "label": clean_name,
                    "category": category,
                    "url": f"/static/sample_images/{file.name}"
                })
    return jsonify({"success": True, "samples": samples})


@app.route("/api/signs/catalog")
def api_signs_catalog():
    """Returns the full recognized road sign dictionary."""
    return jsonify({"success": True, "catalog": SIGN_CATALOG})


@app.route("/api/health")
def api_health():
    """Returns backend system status and diagnostics."""
    import cv2
    return jsonify({
        "status": "online",
        "model": MODEL_NAME,
        "python_version": sys.version,
        "opencv_version": cv2.__version__,
        "database": "sqlite3",
        "auth_methods": ["gmail_otp", "mobile_otp"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })


if __name__ == "__main__":
    logger.info(f"Starting TRAFFIC AI Platform on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
