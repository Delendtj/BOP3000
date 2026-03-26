import cv2
import numpy as np


def select_roi(frame, window_name="ROI Selector"):
    if frame is None:
        return None

    points = []

    def _on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse)

    while True:
        canvas = frame.copy()

        if points:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)
            for px, py in points:
                cv2.circle(canvas, (px, py), 4, (0, 255, 255), -1)

        cv2.putText(
            canvas,
            "Left click: add  Right click/U: undo  C: clear  Enter: save  Esc: cancel",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10) & 0xFF

        if key in (13, 10):
            if len(points) >= 3:
                cv2.destroyWindow(window_name)
                return points
        elif key == 27:
            cv2.destroyWindow(window_name)
            return None
        elif key in (ord("u"), ord("U"), 8):
            if points:
                points.pop()
        elif key in (ord("c"), ord("C")):
            points.clear()


def select_line(frame, window_name="Finish Line"):
    if frame is None:
        return None

    points = []

    def _on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((int(x), int(y)))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _on_mouse)

    while True:
        canvas = frame.copy()

        if points:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(canvas, [pts], False, (0, 165, 255), 2)
            for px, py in points:
                cv2.circle(canvas, (px, py), 5, (0, 165, 255), -1)

        cv2.putText(
            canvas,
            "Left click: add  Right click/U: undo  C: clear  Enter: save  Esc: cancel",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )
        cv2.putText(
            canvas,
            "Lap counts use the line bottom-right point and score left to right.",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )
        cv2.arrowedLine(canvas, (15, 118), (115, 118), (0, 165, 255), 3, tipLength=0.2)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(10) & 0xFF

        if key in (13, 10):
            if len(points) == 2:
                cv2.destroyWindow(window_name)
                return points
        elif key == 27:
            cv2.destroyWindow(window_name)
            return None
        elif key in (ord("u"), ord("U"), 8):
            if points:
                points.pop()
        elif key in (ord("c"), ord("C")):
            points.clear()
