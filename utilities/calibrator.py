import cv2
import numpy as np
import os
import sys

# Allow running this script directly from the repo root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from functions.undistort import undistort

VIDEO_PATH = "../videos/calibration.MP4"
PREVIEW_W, PREVIEW_H = 1920, 1080


def live_tune():
    cap = cv2.VideoCapture(VIDEO_PATH)
    params = {
        'grid_ox': 0, 'grid_oy': 0,
        'step': 5, 'grid': True
    }

    paused = False
    ret, frame = cap.read()

    print("--- UNDISTORT GRID CHECK ---")
    print("ARROWS: Move Grid | ,/.: Step | G: Toggle Grid | Space: Pause | Q: Quit")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        undistorted = undistort(frame)
        if (undistorted.shape[1], undistorted.shape[0]) != (PREVIEW_W, PREVIEW_H):
            undistorted = cv2.resize(
                undistorted,
                (PREVIEW_W, PREVIEW_H),
                interpolation=cv2.INTER_AREA,
            )

        if params['grid']:
            color = (0, 255, 0)
            # Vertical Lines
            for i in range(-10, 11):
                lx = int(PREVIEW_W / 2 + (i * PREVIEW_W / 10) + params['grid_ox'])
                if 0 <= lx < PREVIEW_W:
                    cv2.line(undistorted, (lx, 0), (lx, PREVIEW_H), color, 1)
                # Horizontal Lines
                ly = int(PREVIEW_H / 2 + (i * PREVIEW_H / 10) + params['grid_oy'])
                if 0 <= ly < PREVIEW_H:
                    cv2.line(undistorted, (0, ly), (PREVIEW_W, ly), color, 1)

        cv2.putText(
            undistorted,
            f"Step: {params['step']} | Grid Offset: {params['grid_ox']}, {params['grid_oy']}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Tuner", undistorted)

        # Use waitKeyEx for extended arrow key codes
        key = cv2.waitKeyEx(1)

        if key == -1: continue  # No key pressed

        # Standard Keys (ASCII)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord(' '):
            paused = not paused
        elif key & 0xFF == ord('g'):
            params['grid'] = not params['grid']
        elif key & 0xFF == ord(','):
            params['step'] += 5
        elif key & 0xFF == ord('.'):
            params['step'] = max(1, params['step'] - 5)

        # Robust Arrow Key Handling
        # Windows/Linux often use these codes for waitKeyEx
        elif key == 2490368 or key == 82 or key == 65362:  # Up
            params['grid_oy'] -= params['step']
        elif key == 2621440 or key == 84 or key == 65364:  # Down
            params['grid_oy'] += params['step']
        elif key == 2424832 or key == 81 or key == 65361:  # Left
            params['grid_ox'] -= params['step']
        elif key == 2555904 or key == 83 or key == 65363:  # Right
            params['grid_ox'] += params['step']

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_tune()
