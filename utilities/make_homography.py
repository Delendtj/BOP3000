import os

import cv2, json, numpy as np

DISPLAY_MAX = (1280, 720)


def collect_points(name, image):
    pts = []
    scale = min(DISPLAY_MAX[0] / image.shape[1], DISPLAY_MAX[1] / image.shape[0], 1.0)
    display = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x / scale, y / scale])

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(name, on_click)

    while True:
        vis = display.copy()
        for p in pts:
            px, py = int(p[0] * scale), int(p[1] * scale)
            cv2.circle(vis, (px, py), 5, (0, 255, 0), -1)
        cv2.imshow(name, vis)
        key = cv2.waitKey(10) & 0xFF
        if key in {27, 13}:  # Esc or Enter
            break

    cv2.destroyWindow(name)
    return np.array(pts, dtype=np.float32)

close_path = "../videos/DJI_CUT.MP4"
wide_path = "../videos/canon_1.mp4"

wide_cap = cv2.VideoCapture(wide_path)
_, wide = wide_cap.read()
wide_cap.release()

close_cap = cv2.VideoCapture(close_path)
_, close = close_cap.read()
close_cap.release()

if not isinstance(wide, np.ndarray) or not isinstance(close, np.ndarray):
    raise RuntimeError("Could not read frames from the provided videos.")

wide_pts = collect_points("Wide", wide)
close_pts = collect_points("Close", close)

if len(wide_pts) < 4 or len(close_pts) < 4:
    raise RuntimeError("Need at least four correspondences for a valid homography.")
H, _ = cv2.findHomography(wide_pts, close_pts, cv2.RANSAC, 5.0)
with open("img/homography.json", "w") as f:
  json.dump({"H": H.tolist()}, f)


try:
    HOMOGRAPHY_PATH = os.path.join("img", "homography.json")
    with open(HOMOGRAPHY_PATH, "r") as f:
        print("Homography successfully created")
except FileNotFoundError:
    print("Could not create homography file")
