import json
import os


def load_roi(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "points" in data:
            points = [(int(x), int(y)) for x, y in data["points"]]
            if len(points) >= 3:
                return points
        # Backward-compatible rectangle format.
        if all(k in data for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = (
                int(data["x1"]),
                int(data["y1"]),
                int(data["x2"]),
                int(data["y2"]),
            )
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    return None


def save_roi(path, roi):
    if roi is None:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {"points": [[int(x), int(y)] for x, y in roi]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_line(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "points" not in data:
            return None
        points = [(int(x), int(y)) for x, y in data["points"]]
        if len(points) == 2:
            return points
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    return None


def save_line(path, line):
    if line is None:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {"points": [[int(x), int(y)] for x, y in line]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
