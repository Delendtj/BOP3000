import os
import sys
import argparse
import json

import cv2
import numpy as np

# Allow running this script directly from the repo root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from functions.undistort import undistort
from utilities.downscale_to_1080p import downscale_to_1080p

DISPLAY_MAX = (1280, 720)

# Real-world rink coordinates (in feet or meters, your choice — just be consistent).
# Order here defines the exact click sequence in the UI (top-to-bottom in code).
RINK_POINTS = [
    # Center circle
    ("center ice dot", (0.0, 0.0)),
    ("left center circle", (-4.5, 0.0)),
    ("right center circle", (4.5, 0.0)),
    ("top center circle", (0.0, 4.5)),
    ("bottom center circle", (0.0, -4.5)),

    # Red center line intersections with boards
    ("red line, top board", (0.0, 15.0)),
    ("red line, bottom board", (0.0, -15.0)),

    # Blue line intersections with boards/walls
    ("left blue line, top board", (-7.5, 15.0)),
    #("left blue line, bottom board", (-7.5, -15.0)),
    ("right blue line, top board", (7.5, 15.0)),
    #("right blue line, bottom board", (7.5, -15.0)),

    # Blue line center ice (middle of rink width)
    ("left blue line, center", (-7.5, 0.0)),
    ("right blue line, center", (7.5, 0.0)),
]
WORLD_POINTS = np.array([p for _, p in RINK_POINTS], dtype=np.float32)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create per-camera homographies from image -> rink coordinates."
    )
    parser.add_argument(
        "--wide",
        help="Path to the wide source used as DATA_PATH at runtime.",
    )
    parser.add_argument(
        "--close",
        help="Path to the close source used as CLOSE_SOURCE at runtime.",
    )
    parser.add_argument(
        "--output-wide",
        default="img/homography_wide.json",
        help="Output path for the wide->rink homography JSON.",
    )
    parser.add_argument(
        "--output-close",
        default="img/homography_close.json",
        help="Output path for the close->rink homography JSON.",
    )
    parser.add_argument(
        "--edit",
        help="Path to an existing homography JSON to overwrite (requires exactly one of --wide/--close).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Frame step size when selecting a preview frame.",
    )
    return parser.parse_args()


def collect_points_in_order(name, image, labels, initial_points=None):
    pts = []
    if initial_points is not None:
        pts = [list(p) for p in initial_points]
    mouse_display = [image.shape[1] // 2, image.shape[0] // 2]
    scale = min(DISPLAY_MAX[0] / image.shape[1], DISPLAY_MAX[1] / image.shape[0], 1.0)
    display = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    total = len(labels)
    edit_mode = initial_points is not None and len(pts) == total
    edit_index = 0

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            if edit_mode and len(pts) == total:
                pts[edit_index] = [x / scale, y / scale]
            elif len(pts) < total:
                pts.append([x / scale, y / scale])
        mouse_display[0] = x
        mouse_display[1] = y

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(name, on_mouse)

    while True:
        vis = display.copy()

        for idx, p in enumerate(pts):
            px, py = int(p[0] * scale), int(p[1] * scale)
            color = (0, 255, 0)
            if edit_mode and idx == edit_index:
                color = (0, 255, 255)
            cv2.circle(vis, (px, py), 5, color, -1)
            cv2.putText(
                vis,
                f"{idx + 1}",
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        next_idx = len(pts)
        if edit_mode:
            next_label, next_xy = labels[edit_index]
            prompt = (
                f"Edit {edit_index + 1}/{total}: {next_label} -> {next_xy} | "
                "B=Back, [ / ]=Prev/Next, E=Toggle Edit, Enter=Done, Esc=Cancel"
            )
        elif next_idx < total:
            next_label, next_xy = labels[next_idx]
            prompt = f"Click {next_idx + 1}/{total}: {next_label} -> {next_xy} | B=Back, Enter=Done, Esc=Cancel"
        else:
            prompt = "Done. Press Enter to accept or Esc to cancel."

        cv2.putText(
            vis,
            prompt,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        # Render point list with status.
        list_y = 55
        for idx, (label, coords) in enumerate(labels):
            status = "OK" if idx < len(pts) else "--"
            color = (0, 200, 0) if idx < len(pts) else (200, 200, 200)
            if edit_mode and idx == edit_index:
                color = (0, 255, 255)
            line = f"{idx + 1:02d}. {label} {coords} [{status}]"
            cv2.putText(
                vis,
                line,
                (20, list_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            list_y += 18

        # Zoomed-in inset around cursor.
        zoom_size = 200
        zoom_scale = 3
        img_x = int(mouse_display[0] / scale)
        img_y = int(mouse_display[1] / scale)
        half = max(1, zoom_size // (2 * zoom_scale))
        x1 = max(0, img_x - half)
        y1 = max(0, img_y - half)
        x2 = min(image.shape[1], img_x + half)
        y2 = min(image.shape[0], img_y + half)
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            zoom = cv2.resize(crop, (zoom_size, zoom_size), interpolation=cv2.INTER_NEAREST)
            zx1 = vis.shape[1] - zoom_size - 10
            zy1 = 10
            vis[zy1 : zy1 + zoom_size, zx1 : zx1 + zoom_size] = zoom
            cv2.rectangle(vis, (zx1, zy1), (zx1 + zoom_size, zy1 + zoom_size), (0, 255, 255), 2)
            cv2.line(
                vis,
                (zx1 + zoom_size // 2, zy1),
                (zx1 + zoom_size // 2, zy1 + zoom_size),
                (0, 255, 255),
                1,
            )
            cv2.line(
                vis,
                (zx1, zy1 + zoom_size // 2),
                (zx1 + zoom_size, zy1 + zoom_size // 2),
                (0, 255, 255),
                1,
            )

        cv2.imshow(name, vis)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("e") and len(pts) == total:
            edit_mode = not edit_mode
            continue
        if key == ord("[") and edit_mode:
            edit_index = (edit_index - 1) % total
            continue
        if key == ord("]") and edit_mode:
            edit_index = (edit_index + 1) % total
            continue
        if key == ord("b") and len(pts) > 0 and not edit_mode:
            pts.pop()
            continue
        if key == 27:  # Esc
            cv2.destroyWindow(name)
            return None
        if key == 13 and len(pts) == total:  # Enter
            break

    cv2.destroyWindow(name)
    return np.array(pts, dtype=np.float32)


def select_preview_frame(path: str, window_name: str, step: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {path}.")

    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = None

    def read_frame(index: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if not ret or not isinstance(frame, np.ndarray):
            return None
        return downscale_to_1080p(frame)

    frame = read_frame(frame_idx)
    if frame is None:
        cap.release()
        raise RuntimeError(f"Could not read a frame from {path}.")

    print(
        f"{window_name}: Use Left/Right arrows to change frame by {step}, "
        "Enter to select, Esc to cancel."
    )

    while True:
        vis = frame.copy()
        idx_text = f"Frame {frame_idx + 1}" + (f"/{total}" if total else "")
        cv2.putText(
            vis,
            idx_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.imshow(window_name, vis)
        key = cv2.waitKeyEx(1)
        if key == -1:
            continue
        if key == 27:  # Esc
            cap.release()
            cv2.destroyWindow(window_name)
            return None
        if key == 13:  # Enter
            cap.release()
            cv2.destroyWindow(window_name)
            return frame
        if key in (2555904, 83, 65363):  # Right
            frame_idx += step
            if total is not None:
                frame_idx = min(frame_idx, total - 1)
            frame = read_frame(frame_idx)
        elif key in (2424832, 81, 65361):  # Left
            frame_idx = max(0, frame_idx - step)
            frame = read_frame(frame_idx)


def _load_edit_points(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None
    pts = payload.get("image_points")
    if not pts:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None
    return arr


def _require_edit_points(path: str):
    pts = _load_edit_points(path)
    if pts is None:
        raise SystemExit(
            f"--edit requires '{path}' to contain an 'image_points' list matching RINK_POINTS."
        )
    return pts


def main():
    args = parse_args()
    if not args.wide and not args.close:
        raise SystemExit("Provide at least one source: --wide and/or --close.")
    if args.edit and ((args.wide is None) == (args.close is None)):
        raise SystemExit("When using --edit, provide exactly one of --wide or --close.")
    edit_points = _require_edit_points(args.edit) if args.edit else None

    def reprojection_stats(H, image_pts, world_pts):
        projected = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        diffs = projected - world_pts
        dists = np.linalg.norm(diffs, axis=1)
        return float(dists.mean()), float((dists ** 2).mean() ** 0.5), float(dists.max())

    if args.wide:
        wide_frame = select_preview_frame(args.wide, "Select Wide Frame", step=args.step)
        if wide_frame is None:
            raise RuntimeError("Wide frame selection canceled.")
        wide_frame = downscale_to_1080p(wide_frame)
        wide_frame = undistort(wide_frame)

        wide_pts = collect_points_in_order(
            "Wide (Click in order)",
            wide_frame,
            RINK_POINTS,
            initial_points=edit_points,
        )
        if wide_pts is None or len(wide_pts) != len(WORLD_POINTS):
            raise RuntimeError("Wide point collection canceled or incomplete.")

        wide_H, _ = cv2.findHomography(wide_pts, WORLD_POINTS, cv2.RANSAC, 0.5)
        if wide_H is None:
            raise RuntimeError("Could not estimate wide->rink homography.")

        wide_mean, wide_rms, wide_max = reprojection_stats(wide_H, wide_pts, WORLD_POINTS)
        wide_out = args.edit if args.edit else args.output_wide
        with open(wide_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "H": wide_H.tolist(),
                    "direction": {
                        "source_role": "wide",
                        "target_role": "rink",
                    },
                    "image_points": wide_pts.tolist(),
                },
                f,
                indent=2,
            )
        print(f"Saved wide->rink homography to {wide_out}")
        print(f"Wide reprojection error (rink units): mean={wide_mean:.3f}, rms={wide_rms:.3f}, max={wide_max:.3f}")

    if args.close:
        close_frame = select_preview_frame(args.close, "Select Close Frame", step=args.step)
        if close_frame is None:
            raise RuntimeError("Close frame selection canceled.")
        close_frame = downscale_to_1080p(close_frame)

        close_pts = collect_points_in_order(
            "Close (Click in order)",
            close_frame,
            RINK_POINTS,
            initial_points=edit_points,
        )
        if close_pts is None or len(close_pts) != len(WORLD_POINTS):
            raise RuntimeError("Close point collection canceled or incomplete.")

        close_H, _ = cv2.findHomography(close_pts, WORLD_POINTS, cv2.RANSAC, 0.5)
        if close_H is None:
            raise RuntimeError("Could not estimate close->rink homography.")

        close_mean, close_rms, close_max = reprojection_stats(close_H, close_pts, WORLD_POINTS)
        close_out = args.edit if args.edit else args.output_close
        with open(close_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "H": close_H.tolist(),
                    "direction": {
                        "source_role": "close",
                        "target_role": "rink",
                    },
                    "image_points": close_pts.tolist(),
                },
                f,
                indent=2,
            )
        print(f"Saved close->rink homography to {args.output_close}")
        print(f"Close reprojection error (rink units): mean={close_mean:.3f}, rms={close_rms:.3f}, max={close_max:.3f}")


if __name__ == "__main__":
    main()
