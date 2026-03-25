import argparse
import json
import os
import sys

import cv2
import numpy as np

# Allow running this script directly from the repo root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from functions.spatial.undistort import undistort
from utilities.downscale_to_1080p import downscale_to_1080p
from utilities.make_homography import RINK_POINTS, WORLD_POINTS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify image->rink homography by projecting rink points back into image space."
    )
    parser.add_argument("--video", required=True, help="Path to source video.")
    parser.add_argument("--homography", required=True, help="Path to homography JSON.")
    return parser.parse_args()


def load_frame(path: str) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    cap.release()
    if not ret or not isinstance(frame, np.ndarray):
        raise RuntimeError(f"Could not read a frame from {path}.")
    return downscale_to_1080p(frame)


def load_homography(path: str) -> tuple[np.ndarray, str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    matrix = np.asarray(payload["H"], dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected 3x3 homography in {path}, got {matrix.shape}.")
    source_role = str(payload.get("direction", {}).get("source_role", ""))
    return matrix, source_role


def main():
    args = parse_args()

    frame = load_frame(args.video)

    H, source_role = load_homography(args.homography)
    if source_role == "wide":
        frame = undistort(frame)
    H_inv = np.linalg.inv(H)

    world_pts = WORLD_POINTS.reshape(-1, 1, 2)
    img_pts = cv2.perspectiveTransform(world_pts, H_inv).reshape(-1, 2)

    vis = frame.copy()
    for idx, ((label, _), (x, y)) in enumerate(zip(RINK_POINTS, img_pts), start=1):
        px, py = int(x), int(y)
        if 0 <= px < vis.shape[1] and 0 <= py < vis.shape[0]:
            cv2.circle(vis, (px, py), 6, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"{idx}",
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Homography Check", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
