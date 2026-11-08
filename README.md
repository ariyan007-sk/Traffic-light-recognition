# TRAFFIC AI – Intelligent Traffic Light & Road Sign Recognition System
### AI-Powered Traffic Signal & Road Sign Detection &bull; Project-Based Learning (PBL) in Artificial Intelligence

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Flask%203.x-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![AI Model](https://img.shields.io/badge/YOLO-Ultralytics%20YOLO11n-00ffff?logo=yolo)](https://docs.ultralytics.com/)
[![Computer Vision](https://img.shields.io/badge/CV-OpenCV%20HSV-red?logo=opencv)](https://opencv.org/)
[![Database](https://img.shields.io/badge/Database-SQLite3-green?logo=sqlite)](https://sqlite.org/)
[![Authentication](https://img.shields.io/badge/Auth-Gmail%20%26%20Mobile%20OTP-purple)](https://flask.palletsprojects.com/)

---

## 1. Project Overview

**TRAFFIC AI** is an intelligent computer vision application built for College Project-Based Learning (PBL) in Artificial Intelligence. The system automatically detects physical traffic lights and standard road signs, determines their state, and provides real-time driving guidance:

### 🚦 Traffic Signals
* 🔴 **RED** &rarr; **STOP** (Vehicles must come to a complete halt)
* 🟡 **YELLOW** &rarr; **CAUTION** (Prepare to stop safely)
* 🟢 **GREEN** &rarr; **GO** (Intersection clear, proceed with care)
* ⚪ **UNKNOWN / OFF** &rarr; **INDETERMINATE** (Signal unlit, obscured, or ambiguous)

### 🚸 Road Signs (Mandatory, Warning, Prohibitory & Informational)
* 🛑 **Stop Sign** &rarr; Mandatory full stop before stop line.
* 🔻 **Give Way / Yield** &rarr; Yield right-of-way to oncoming traffic.
* 3️⃣0️⃣ **Speed Limit 30 km/h** &rarr; Maximum speed 30 km/h (school/residential zone).
* 5️⃣0️⃣ **Speed Limit 50 km/h** &rarr; Standard urban speed limit.
* ↰ **Turn Left Ahead** &rarr; Mandatory left turn.
* ↱ **Turn Right Ahead** &rarr; Mandatory right turn.
* 🚫 **No Left Turn / No Right Turn** &rarr; Turning prohibited.
* 🅿️🚫 **No Parking** &rarr; Parking strictly prohibited.
* 🔄 **Roundabout Ahead** &rarr; Circulate clockwise, give way to right.
* 🚧 **Road Work Ahead** &rarr; Caution: Construction workers on road.
* ⚠️ **Speed Breaker / Hump** &rarr; Reduce speed for road cushion.
* ➕ **Crossroads Ahead** &rarr; Approaching 4-way intersection.
* 🅿️ **Parking Area** &rarr; Designated public parking.
* ⬆️ **One-Way Traffic** &rarr; Travel in arrow direction only.

### 🔐 User Authentication (Gmail & Mobile OTP)
* Secure login via **Gmail / Email** or **Mobile Phone Number**.
* 6-digit cryptographic **One-Time Password (OTP)** verification.
* Operator session tracking linked to detection audit history.
* Built-in presentation mode with OTP quick-fill for college evaluations.

---

## 2. Technical Architecture

```text
+-------------------------------------------------------------+
|                      INPUT ACQUISITION                      |
|       Static Image File (JPG/PNG) OR Live Webcam Video       |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|                  STAGE 1: OBJECT LOCALIZATION               |
|      - YOLO11n Target Classes: 'traffic light' & 'stop sign'│
|      - CV Chromatic Contour Scanner (Red, Blue, Yellow)     |
|      - Outputs: Bounding Box Coordinates [x1, y1, x2, y2]   |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|              STAGE 2: ROI CROPPING & ISOLATION              |
|        Sub-image Slicing: roi = frame[y1:y2, x1:x2]         |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|            STAGE 3: HSV & GEOMETRIC CLASSIFICATION          |
|   1. Traffic Light: Dual-Band Red, Yellow, Green HSV        |
|      + Top/Middle/Bottom Spatial Zone Verification          |
|   2. Road Sign: Shape analysis (Circle, Triangle, Octagon,  |
|      Rectangle) + Inner symbol moment / silhouette analysis |
+-------------------------------------------------------------+
                              │
                              ▼
+-------------------------------------------------------------+
|             STAGE 4: TELEMETRY & PRESENTATION               |
|   - Driver Action: STOP, CAUTION, GO, MAX 30, TURN LEFT     |
|   - Sleek OpenCV HUD Box Overlays & Cyber Corner Brackets   |
|   - Real-time 3D Physical Signal Widget Simulation          |
|   - SQLite Historical Audit Logging (Linked to User ID)     |
+-------------------------------------------------------------+
```

---

## 3. Project File Structure

```text
traffic_light_ai/
│
├── api/
│   └── index.py                # Serverless Entry Point (Vercel / AWS Lambda)
├── app.py                      # Flask Application, Auth Routes & REST Gateway
├── config.py                   # Centralized Configuration & HSV Thresholds
├── vercel.json                 # Vercel Serverless Deployment Configuration
├── requirements.txt            # Ultra-lightweight dependencies (<100 MB Bundle)
├── requirements-full.txt       # Optional PyTorch/CUDA dependencies for local training
├── README.md                   # Complete Documentation & Teacher Viva Guide
├── generate_samples.py         # Sample Generator Script for Test Scenarios
│
├── models/
│   ├── yolo11n.onnx            # Lightweight ONNX Runtime Model (10.7 MB)
│   └── yolo11n.pt              # Ultralytics Weights (5.6 MB)
│
├── services/
│   ├── detection.py            # Unified Traffic Light & Road Sign Detector
│   ├── onnx_detector.py        # Ultra-lightweight ONNX Runtime inference engine
│   ├── color_classifier.py     # HSV Colorimetry & Spatial Verification
│   ├── sign_classifier.py      # Road Sign Geometric & Feature Classifier
│   ├── auth_service.py         # Gmail & Mobile OTP Authentication Service
│   └── history.py              # History & Metrics Service
│
├── database/
│   ├── db.py                   # SQLite Connection, Users & OTP Schema
│   └── traffic_history.db      # SQLite Audit & Credential Database
│
├── templates/
│   ├── base.html               # Master Layout, User Avatar & Theme Toggle
│   ├── dashboard.html          # Main Overview, 3D Signal & Road Signs Stat Cards
│   ├── image_detect.html       # Image Upload, Road Sign Gallery & Download
│   ├── webcam_detect.html      # Real-Time Webcam WebRTC Video Module
│   ├── history.html            # Audit Logs with Category Filter & CSV Export
│   ├── about.html              # Academic PBL Report for College Submission
│   └── login.html              # Gmail & Mobile OTP Sign-In Page
│
├── static/
│   ├── css/
│   │   └── style.css           # Dark Navy Smart-City UI, OTP Digits & Neon Glow
│   ├── js/
│   │   ├── app.js              # Theme Switcher, Modals & Toast Alerts
│   │   └── webcam.js           # Live WebRTC Frame Capture & Overlay HUD
│   └── sample_images/          # Built-in Test Images (Lights & Road Signs)
│
└── tests/
    └── test_detection.py       # 14 Automated Unit & Integration Tests
```

---

## 4. Installation & Running Instructions

### Step 1: Open in VS Code
Open VS Code and choose **File &rarr; Open Folder...**:
```text
C:\Users\sariy\.gemini\antigravity\scratch\traffic_light_ai
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Run the Application
```powershell
python app.py
```
Output:
```text
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Step 4: Open in Browser
Navigate to **`http://127.0.0.1:5000`** in your browser.

---

## 5. Serverless & Cloud Deployment (<100 MB Bundle vs 5.8 GB Limit)

If deploying to **Vercel**, **AWS Lambda**, or **Render**, serverless platforms enforce a **500 MB maximum uncompressed bundle limit**.

### Why did standard PyTorch fail?
Standard `torch` + `torchvision` wheels bundled with CUDA 12 libraries (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `triton`, etc.) download **5,806.25 MB** of files, causing the serverless build to immediately fail with:
> `Total bundle size (5806.25 MB) exceeds the maximum function size (500 MB).`

### How we solved it:
1. **ONNX Runtime CPU Engine (`onnxruntime`)**: We converted the YOLO11n weights to ONNX (`models/yolo11n.onnx`, only 10.7 MB).
2. **Headless OpenCV (`opencv-python-headless`)**: Eliminates heavy GUI/X11 libraries.
3. **Bundle Size Reduction**:
   - PyTorch + CUDA: **5,806.25 MB** (❌ Over limit)
   - Production ONNX Bundle: **~95 MB** (✅ **98.4% reduction**, well below 500 MB!)
4. **Vercel Config Included**: `vercel.json` and `api/index.py` are pre-configured to deploy in 1 click.

---

## 6. Automated Test Results (14/14 Passed)

To execute the test suite:
```powershell
python tests/test_detection.py
```

Output:
```text
test_gmail_otp_lifecycle (__main__.TestAuthService) ... ok
test_mobile_otp_lifecycle (__main__.TestAuthService) ... ok
test_validate_identifiers (__main__.TestAuthService) ... ok
test_green_roi_detection (__main__.TestColorClassifier) ... ok
test_red_roi_detection (__main__.TestColorClassifier) ... ok
test_unlit_roi_returns_unknown (__main__.TestColorClassifier) ... ok
test_yellow_roi_detection (__main__.TestColorClassifier) ... ok
test_api_signs_catalog_endpoint (__main__.TestFlaskEndpoints) ... ok
test_auth_endpoints_flow (__main__.TestFlaskEndpoints) ... ok
test_page_routes_return_200 (__main__.TestFlaskEndpoints) ... ok
test_give_way_detection (__main__.TestRoadSignClassifier) ... ok
test_speed_limit_30_detection (__main__.TestRoadSignClassifier) ... ok
test_stop_sign_detection (__main__.TestRoadSignClassifier) ... ok
test_turn_left_ahead_detection (__main__.TestRoadSignClassifier) ... ok

----------------------------------------------------------------------
Ran 14 tests in 10.048s

OK
```

---

## 6. How to Demonstrate to Your College Teacher

### 1. Operator Sign In via Gmail / Mobile OTP
1. Click **"Sign In"** in the top-right navbar.
2. Select **"Gmail Address"** &rarr; enter `officer@gmail.com` (or choose **"Mobile Number"** &rarr; `9876543210`).
3. Click **"Send Verification Code (OTP)"**.
4. Observe the 6-digit OTP code displayed in the demo notification.
5. Click **"Auto-Fill"** (or type the digits) and click **"Verify & Sign In"**.
6. The user avatar appears in the top-right navbar with their name and verified status!

### 2. Road Signs Recognition
1. Navigate to **Image Analysis**.
2. Click the **"Road Signs"** filter pill in the Quick Test Gallery.
3. Test **"Sample Sign Stop"** &rarr; Recognized as `Stop Sign (STOP)`.
4. Test **"Sample Sign Speed 30"** &rarr; Recognized as `Speed Limit 30 km/h (MAX 30 KM/H)`.
5. Test **"Sample Sign Turn Left"** &rarr; Recognized as `Turn Left Ahead (TURN LEFT)`.
6. Test **"Sample Sign Turn Right"** &rarr; Recognized as `Turn Right Ahead (TURN RIGHT)`.
7. Test **"Sample Sign Give Way"** &rarr; Recognized as `Give Way / Yield (YIELD)`.
8. Test **"Sample Sign Road Work"** &rarr; Recognized as `Road Work Ahead (CAUTION - WORKERS)`.
9. Test **"Sample Sign Hump"** &rarr; Recognized as `Speed Breaker / Hump (SLOW DOWN)`.
10. Test **"Sample Sign Parking"** &rarr; Recognized as `Parking Area (PARKING)`.

### 3. Traffic Lights Recognition
1. Click the **"Traffic Lights"** filter pill.
2. Test Red, Yellow, and Green lights and observe the 3D signal widget.

### 4. Audit History & SQLite Records
1. Open **History Logs**.
2. Notice the detections are tagged with the active Operator ID (`officer@gmail.com`).
3. Filter by **"Road Signs"** or **"Red Signals"** and click **Export CSV**.
