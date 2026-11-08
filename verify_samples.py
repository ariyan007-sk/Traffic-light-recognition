import cv2
from pathlib import Path
from services.detection import detect_traffic_lights

sample_dir = Path("static/sample_images")
for p in sorted(sample_dir.glob("sample_sign_*.jpg")):
    img = cv2.imread(str(p))
    res = detect_traffic_lights(img)
    print(f"{p.name}: Detected={res['detected']}, Signal={res['primary_signal']}, Inst={res['primary_instruction']}, Category={res.get('category')}, Conf={res['color_confidence']}%")
