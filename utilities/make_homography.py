import os
import sys
import argparse
import json

import cv2
import numpy as np

from downscale_to_1080p import downscale_to_1080p

# Allow running this script directly from the repo root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from functions.undistort import undistort

DISPLAY_MAX = (1280, 720)

# Real-world rink coordinates (in feet or meters, your choice — just be consistent)
# These are the known positions of markings on the rink
world_points = np.array([
    [0.0,   0.0],   # e.g. center ice dot
    [25.0,  0.0],   # e.g. right blue line center
    [-25.0, 0.0],   # e.g. left blue line center
    [25.0,  22.0],  # e.g. right face-off dot
    [25.0, -22.0],  # e.g. right face-off dot
    [-25.0, 22.0],  # etc.
    [-25.0, -22.0],
], dtype=np.float32)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a wide-to-close homography for the multi-camera matcher."
    )
    parser.add_argument(
        "--wide",
        required=True,
        help="Path to the wide source used as DATA_PATH at runtime.",
    )
    parser.add_argument(
        "--close",
        required=True,
        help="Path to the close source used as CLOSE_SOURCE at runtime.",
    )
    parser.add_argument(
        "--output",
        default="img/homography.json",
        help="Output path for the wide-to-close homography JSON.",
    )
    return parser.parse_args()


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


def load_preview_frame(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()

    if not ret or not isinstance(frame, np.ndarray):
        raise RuntimeError(f"Could not read a frame from {path}.")

    return downscale_to_1080p(frame)


def main():
    args = parse_args()

    wide_frame = load_preview_frame(args.wide)
    close_frame = load_preview_frame(args.close)

    # Downscale
    wide_frame = downscale_to_1080p(wide_frame)
    close_frame = downscale_to_1080p(close_frame)

    # Undistorts the wide lens before setting points
    wide_frame = undistort(wide_frame)

    wide_pts = collect_points("Wide", wide_frame)
    close_pts = collect_points("Close", close_frame)

    if len(wide_pts) < 4 or len(close_pts) < 4:
        raise RuntimeError("Need at least four correspondences for a valid homography.")

    # Build homography in the direction expected at runtime: close -> wide.
    H, _ = cv2.findHomography(close_pts, wide_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Could not estimate a homography from the selected points.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "H": H.tolist(),
                "direction": {
                    "source_role": "close",
                    "target_role": "wide",
                },
            },
            f,
            indent=2,
        )

    print(f"Saved close->wide homography to {args.output}")


if __name__ == "__main__":
    main()
