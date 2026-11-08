"""
Configuration settings for TRAFFIC AI - Intelligent Traffic Light Recognition System.
Designed for College Project-Based Learning (PBL) in Artificial Intelligence.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
SAMPLE_DIR = STATIC_DIR / "sample_images"

# Ensure runtime directories exist
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Serverless Environment Detection (e.g., Vercel, AWS Lambda)
IS_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))

if IS_SERVERLESS:
    # Serverless runtime requires writeable directory at /tmp
    DATABASE_PATH = Path("/tmp") / "traffic_history.db"
    UPLOAD_DIR = Path("/tmp") / "uploads"
else:
    DATABASE_PATH = DATABASE_DIR / "traffic_history.db"
    UPLOAD_DIR = STATIC_DIR / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# AI Model Configuration
# YOLO11n compact real-time model
MODEL_NAME = "yolo11n.pt"
MODEL_PATH = MODELS_DIR / MODEL_NAME
ONNX_MODEL_NAME = "yolo11n.onnx"
ONNX_MODEL_PATH = MODELS_DIR / ONNX_MODEL_NAME
PREFER_ONNX = True  # True enables ultra-lightweight ONNX Runtime (~90 MB bundle) vs PyTorch (~5.8 GB)
CONFIDENCE_THRESHOLD = 0.35
COCO_TRAFFIC_LIGHT_CLASS_ID = 9  # COCO dataset class index for 'traffic light'
COCO_STOP_SIGN_CLASS_ID = 11      # COCO dataset class index for 'stop sign'

# Image Processing & HSV Color Ranges
# Traffic signals use distinctive hues; HSV color space isolates illumination from intensity.
HSV_COLOR_RANGES = {
    "RED": [
        # Red spans the 0/180 hue boundary in OpenCV's 0-180 scale
        {"lower": (0, 70, 110), "upper": (10, 255, 255)},
        {"lower": (160, 70, 110), "upper": (180, 255, 255)}
    ],
    "YELLOW": [
        # Amber / Yellow signal range
        {"lower": (12, 80, 110), "upper": (36, 255, 255)}
    ],
    "GREEN": [
        # Green / Cyan-green traffic signal range
        {"lower": (38, 50, 90), "upper": (95, 255, 255)}
    ]
}

# Signal Instruction Mapping
SIGNAL_INSTRUCTIONS = {
    "RED": {
        "instruction": "STOP",
        "description": "Red Signal Active. Vehicles must come to a complete halt behind the stop line.",
        "badge_color": "danger"
    },
    "YELLOW": {
        "instruction": "CAUTION",
        "description": "Yellow Signal Active. Prepare to stop unless it is unsafe to do so.",
        "badge_color": "warning"
    },
    "GREEN": {
        "instruction": "GO",
        "description": "Green Signal Active. Proceed through the intersection with care.",
        "badge_color": "success"
    },
    "UNKNOWN": {
        "instruction": "INDETERMINATE",
        "description": "Signal state unclear or light is unlit/obscured. Exercise caution.",
        "badge_color": "secondary"
    }
}

# Web Server Configuration
HOST = "0.0.0.0"  # Binds to all network interfaces for local network & tunnel access
PORT = 5000
DEBUG = True
SECRET_KEY = "traffic-light-ai-secret-key-pbl-2026"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
