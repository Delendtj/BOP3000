import cv2
import numpy as np

VIDEO_PATH = "DJI25.MP4"
PREVIEW_W, PREVIEW_H = 1920, 1080


def build_stereographic_vectorized(pano_w, pano_h, fx, fy, cx, cy):
    u, v = np.meshgrid(np.arange(pano_w), np.arange(pano_h))
    xn = (u - pano_w / 2) / (pano_w / 2)
    yn = (v - pano_h / 2) / (pano_h / 2)
    d = xn ** 2 + yn ** 2
    x = 2 * xn / (1 + d)
    y = 2 * yn / (1 + d)
    z = (1 - d) / (1 + d)
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(r, z)
    mask = r > 1e-6
    scale = np.zeros_like(r)
    scale[mask] = theta[mask] / r[mask]
    map_x = cx + (x * scale) * fx
    map_y = cy + (y * scale) * fy
    return map_x.astype(np.float32), map_y.astype(np.float32)


def live_tune():
    cap = cv2.VideoCapture(VIDEO_PATH)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    params = {
        'fx': 1000, 'fy': 800,
        'cx': src_w // 2, 'cy': src_h // 2,
        'grid_ox': 0, 'grid_oy': 0,
        'step': 5, 'grid': True
    }

    paused = False
    ret, frame = cap.read()

    print("--- ROBUST KEYBOARD TUNING ---")
    print("W/S: FY | A/D: FX | ARROWS: Move Grid | ,/.: Step | Q: Quit")

    while True:
        m1, m2 = build_stereographic_vectorized(PREVIEW_W, PREVIEW_H,
                                                params['fx'], params['fy'],
                                                params['cx'], params['cy'])

        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        undistorted = cv2.remap(frame, m1, m2, cv2.INTER_LINEAR)

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

        cv2.putText(undistorted, f"FX: {params['fx']} FY: {params['fy']} Step: {params['step']}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(undistorted, f"Grid Offset: {params['grid_ox']}, {params['grid_oy']}",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
        elif key & 0xFF == ord('w'):
            params['fy'] += params['step']
        elif key & 0xFF == ord('s'):
            params['fy'] -= params['step']
        elif key & 0xFF == ord('d'):
            params['fx'] += params['step']
        elif key & 0xFF == ord('a'):
            params['fx'] -= params['step']
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