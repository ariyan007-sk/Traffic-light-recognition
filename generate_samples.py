"""
Generates high-fidelity traffic scene test images for TRAFFIC AI demonstration.
Includes Red, Yellow, Green Traffic Lights, plus Road Signs from the Standard Catalog:
Stop, Speed Limit 30, Turn Left, Turn Right, No Parking, Road Work, Hump, Give Way, Parking.
"""

import cv2
import numpy as np
from pathlib import Path

SAMPLE_DIR = Path("C:/Users/sariy/.gemini/antigravity/scratch/traffic_light_ai/static/sample_images")
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)


def draw_realistic_traffic_scene(active_color: str, filename: str):
    """Draws an urban intersection scene with a clear traffic light fixture."""
    w, h = 640, 480
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Sky Gradient
    for y in range(int(h * 0.65)):
        ratio = y / (h * 0.65)
        img[y, :] = (int(140 + ratio * 60), int(110 + ratio * 50), int(70 + ratio * 40))

    # 2. Distant Buildings
    np.random.seed(42)
    for bx in range(0, w, 70):
        bh = np.random.randint(140, 220)
        bw = np.random.randint(60, 90)
        cv2.rectangle(img, (bx, int(h * 0.65) - bh), (bx + bw, int(h * 0.65)), (75, 80, 85), -1)

    # 3. Ground / Road Surface
    road_y = int(h * 0.65)
    for y in range(road_y, h):
        ratio = (y - road_y) / (h - road_y)
        asphalt = int(45 + ratio * 20)
        img[y, :] = (asphalt, asphalt, asphalt)

    # Sidewalk curb
    cv2.line(img, (0, road_y), (w, road_y), (100, 105, 110), 4)

    # 4. Traffic Pole & Cantilever Arm
    pole_x = int(w * 0.72)
    pole_w = 14
    cv2.rectangle(img, (pole_x, int(h * 0.2)), (pole_x + pole_w, road_y + 10), (70, 75, 80), -1)
    arm_y = int(h * 0.22)
    cv2.rectangle(img, (int(w * 0.40), arm_y), (pole_x + pole_w, arm_y + 12), (70, 75, 80), -1)

    # 5. Traffic Light Housing
    tl_w, tl_h = 64, 170
    tl_x = int(w * 0.44)
    tl_y = arm_y + 8

    cv2.rectangle(img, (tl_x - 6, tl_y - 6), (tl_x + tl_w + 6, tl_y + tl_h + 6), (15, 18, 22), -1)
    cv2.rectangle(img, (tl_x, tl_y), (tl_x + tl_w, tl_y + tl_h), (25, 28, 33), -1)

    lamp_r = 20
    cx = tl_x + tl_w // 2
    y_red, y_yel, y_grn = tl_y + 32, tl_y + 85, tl_y + 138

    # Visors & Base unlit lamps
    for cy in [y_red, y_yel, y_grn]:
        cv2.ellipse(img, (cx, cy - lamp_r + 2), (lamp_r + 4, 10), 0, 180, 360, (15, 18, 22), -1)
        cv2.circle(img, (cx, cy), lamp_r, (35, 38, 45), -1)

    # Illuminated Lamp
    if active_color == "RED":
        cv2.circle(img, (cx, y_red), lamp_r + 8, (0, 0, 180), -1)
        cv2.circle(img, (cx, y_red), lamp_r, (30, 30, 255), -1)
        cv2.circle(img, (cx - 4, y_red - 4), int(lamp_r * 0.45), (140, 140, 255), -1)
    elif active_color == "YELLOW":
        cv2.circle(img, (cx, y_yel), lamp_r + 8, (0, 160, 220), -1)
        cv2.circle(img, (cx, y_yel), lamp_r, (0, 215, 255), -1)
        cv2.circle(img, (cx - 4, y_yel - 4), int(lamp_r * 0.45), (120, 240, 255), -1)
    elif active_color == "GREEN":
        cv2.circle(img, (cx, y_grn), lamp_r + 8, (0, 180, 80), -1)
        cv2.circle(img, (cx, y_grn), lamp_r, (30, 255, 100), -1)
        cv2.circle(img, (cx - 4, y_grn - 4), int(lamp_r * 0.45), (140, 255, 180), -1)

    cv2.imwrite(str(SAMPLE_DIR / filename), img)
    print(f"Generated {filename}")


def draw_road_sign_scene(sign_type: str, filename: str):
    """Draws a roadside scene featuring a specific road sign on a metallic sign post."""
    w, h = 640, 480
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Road background (pale natural daylight with low saturation so it doesn't trigger color masks)
    for y in range(int(h * 0.65)):
        img[y, :] = (235, 225, 215)
    road_y = int(h * 0.65)
    for y in range(road_y, h):
        img[y, :] = (55, 55, 55)

    # Sidewalk grass & trees
    cv2.rectangle(img, (0, road_y - 20), (w, road_y), (40, 120, 60), -1)

    # Metal sign post
    post_x = int(w * 0.5) - 6
    cv2.rectangle(img, (post_x, int(h * 0.2)), (post_x + 12, road_y + 10), (120, 125, 130), -1)

    # Sign center
    scx, scy = int(w * 0.5), int(h * 0.35)
    r = 65

    if sign_type == "STOP":
        # Octagon
        pts = []
        for a in range(8):
            angle = (a * 45 + 22.5) * np.pi / 180
            pts.append([int(scx + r * np.cos(angle)), int(scy + r * np.sin(angle))])
        pts = np.array([pts], np.int32)
        cv2.fillPoly(img, pts, (30, 30, 220))
        cv2.polylines(img, pts, True, (255, 255, 255), 4)
        # STOP text
        cv2.putText(img, "STOP", (scx - 48, scy + 14), cv2.FONT_HERSHEY_DUPLEX, 1.25, (255, 255, 255), 3, cv2.LINE_AA)

    elif sign_type == "SPEED_30":
        # Red circle, white center
        cv2.circle(img, (scx, scy), r, (30, 30, 220), -1)
        cv2.circle(img, (scx, scy), r - 12, (255, 255, 255), -1)
        cv2.putText(img, "30", (scx - 30, scy + 16), cv2.FONT_HERSHEY_DUPLEX, 1.4, (20, 20, 20), 4, cv2.LINE_AA)

    elif sign_type == "TURN_LEFT":
        # Blue circle, white arrow turning left
        cv2.circle(img, (scx, scy), r, (220, 100, 20), -1)
        cv2.circle(img, (scx, scy), r - 4, (255, 255, 255), 2)
        # Left turn arrow
        pts = np.array([[scx + 15, scy + 35], [scx + 15, scy - 5], [scx - 15, scy - 5],
                        [scx - 15, scy - 20], [scx - 42, scy + 5], [scx - 15, scy + 30],
                        [scx - 15, scy + 15], [scx + 5, scy + 15], [scx + 5, scy + 35]], np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255))

    elif sign_type == "TURN_RIGHT":
        # Blue circle, white arrow turning right
        cv2.circle(img, (scx, scy), r, (220, 100, 20), -1)
        cv2.circle(img, (scx, scy), r - 4, (255, 255, 255), 2)
        # Right turn arrow
        pts = np.array([[scx - 15, scy + 35], [scx - 15, scy - 5], [scx + 15, scy - 5],
                        [scx + 15, scy - 20], [scx + 42, scy + 5], [scx + 15, scy + 30],
                        [scx + 15, scy + 15], [scx - 5, scy + 15], [scx - 5, scy + 35]], np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255))

    elif sign_type == "NO_PARKING":
        # Red circle, blue center with red diagonal slash
        cv2.circle(img, (scx, scy), r, (30, 30, 220), -1)
        cv2.circle(img, (scx, scy), r - 12, (220, 110, 20), -1)
        # Diagonal red slash
        cv2.line(img, (int(scx - r * 0.7), int(scy - r * 0.7)), (int(scx + r * 0.7), int(scy + r * 0.7)), (30, 30, 220), 12)

    elif sign_type == "HUMP":
        # Red triangle warning
        pts = np.array([[scx, scy - r], [scx - r - 8, scy + r - 8], [scx + r + 8, scy + r - 8]], np.int32)
        cv2.fillPoly(img, [pts], (30, 30, 220))
        pts_in = np.array([[scx, scy - r + 16], [scx - r + 8, scy + r - 16], [scx + r - 8, scy + r - 16]], np.int32)
        cv2.fillPoly(img, [pts_in], (255, 255, 255))
        # Hump curve
        cv2.ellipse(img, (scx, scy + 18), (28, 14), 0, 180, 360, (20, 20, 20), -1)

    elif sign_type == "ROAD_WORK":
        # Red triangle warning
        pts = np.array([[scx, scy - r], [scx - r - 8, scy + r - 8], [scx + r + 8, scy + r - 8]], np.int32)
        cv2.fillPoly(img, [pts], (30, 30, 220))
        pts_in = np.array([[scx, scy - r + 16], [scx - r + 8, scy + r - 16], [scx + r - 8, scy + r - 16]], np.int32)
        cv2.fillPoly(img, [pts_in], (255, 255, 255))
        # Silhouette
        cv2.circle(img, (scx, scy - 4), 8, (20, 20, 20), -1)
        cv2.line(img, (scx, scy + 4), (scx + 12, scy + 24), (20, 20, 20), 5)
        cv2.line(img, (scx + 8, scy + 18), (scx - 14, scy + 30), (20, 20, 20), 4)

    elif sign_type == "GIVE_WAY":
        # Inverted triangle
        pts = np.array([[scx - r - 8, scy - r + 8], [scx + r + 8, scy - r + 8], [scx, scy + r]], np.int32)
        cv2.fillPoly(img, [pts], (30, 30, 220))
        pts_in = np.array([[scx - r + 8, scy - r + 18], [scx + r - 8, scy - r + 18], [scx, scy + r - 14]], np.int32)
        cv2.fillPoly(img, [pts_in], (255, 255, 255))

    elif sign_type == "PARKING":
        # Blue square with P
        cv2.rectangle(img, (scx - r, scy - r), (scx + r, scy + r), (220, 110, 20), -1)
        cv2.rectangle(img, (scx - r + 4, scy - r + 4), (scx + r - 4, scy + r - 4), (255, 255, 255), 2)
        cv2.putText(img, "P", (scx - 22, scy + 24), cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 255, 255), 5, cv2.LINE_AA)

    cv2.imwrite(str(SAMPLE_DIR / filename), img)
    print(f"Generated {filename}")


def draw_no_traffic_light_scene(filename: str):
    """Draws a road scene without any traffic signals or signs."""
    w, h = 640, 480
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(int(h * 0.6)):
        img[y, :] = (210, 180, 130)
    for y in range(int(h * 0.6), h):
        img[y, :] = (55, 55, 55)
    for tx in [40, 100, 520, 580]:
        cv2.rectangle(img, (tx, int(h * 0.48)), (tx + 12, int(h * 0.6)), (30, 50, 70), -1)
        cv2.circle(img, (tx + 6, int(h * 0.45)), 35, (35, 95, 45), -1)
    cv2.imwrite(str(SAMPLE_DIR / filename), img)
    print(f"Generated {filename}")


if __name__ == "__main__":
    # Traffic Lights
    draw_realistic_traffic_scene("RED", "sample_red_signal.jpg")
    draw_realistic_traffic_scene("YELLOW", "sample_yellow_signal.jpg")
    draw_realistic_traffic_scene("GREEN", "sample_green_signal.jpg")
    draw_no_traffic_light_scene("sample_no_traffic_light.jpg")

    # Road Signs (from user's picture)
    draw_road_sign_scene("STOP", "sample_sign_stop.jpg")
    draw_road_sign_scene("SPEED_30", "sample_sign_speed_30.jpg")
    draw_road_sign_scene("TURN_LEFT", "sample_sign_turn_left.jpg")
    draw_road_sign_scene("TURN_RIGHT", "sample_sign_turn_right.jpg")
    draw_road_sign_scene("NO_PARKING", "sample_sign_no_parking.jpg")
    draw_road_sign_scene("HUMP", "sample_sign_hump.jpg")
    draw_road_sign_scene("ROAD_WORK", "sample_sign_road_work.jpg")
    draw_road_sign_scene("GIVE_WAY", "sample_sign_give_way.jpg")
    draw_road_sign_scene("PARKING", "sample_sign_parking.jpg")

    print("All traffic lights and road signs generated successfully!")
