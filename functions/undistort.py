import numpy as np
import cv2

DIM = (3840, 2160)  # (width, height) used during calibration

K = np.array(
    [
        [969.4786843037966, 0.0, 1923.8393363075347],
        [0.0, 946.927038666443, 1078.8186005266996],
        [0.0, 0.0, 1.0],
    ]
)
D = np.array(
    [
        [0.30933055236964674],
        [-0.12206780204557209],
        [0.13870483068957873],
        [-0.047928667768809004],
    ]
)


def _scaled_K(img_shape):
    """
    Scale the calibrated K to the current image size.
    img_shape: (h, w, c) from a numpy image.
    """
    if img_shape is None:
        return K
    h, w = img_shape[:2]
    if (w, h) == DIM:
        return K
    sx = w / float(DIM[0])
    sy = h / float(DIM[1])
    k = K.copy()
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] *= sx
    k[1, 2] *= sy
    return k

def undistort(img):
    """
    Returns undistorted fisheye/wide-lens image based on vars above.
    """

    k = _scaled_K(img.shape)
    dim = (img.shape[1], img.shape[0])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        k, D, np.eye(3), k, dim, cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return undistorted_img


def undistort_points(points, img_shape=None):
    """
    Undistort pixel points into the undistorted image space.
    points: iterable of (x, y) in distorted pixel coordinates.
    Returns Nx2 array of undistorted pixel coordinates.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    k = _scaled_K(img_shape)
    undist = cv2.fisheye.undistortPoints(pts, k, D, P=k)
    return undist.reshape(-1, 2)


def distort_points(points, img_shape=None):
    """
    Distort undistorted pixel points back into the distorted wide image space.
    points: iterable of (x, y) in undistorted pixel coordinates.
    Returns Nx2 array of distorted pixel coordinates.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    k = _scaled_K(img_shape)

    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]
    normalized = np.empty((pts.shape[0], 1, 2), dtype=np.float32)
    normalized[:, 0, 0] = (pts[:, 0] - cx) / fx
    normalized[:, 0, 1] = (pts[:, 1] - cy) / fy

    distorted = cv2.fisheye.distortPoints(normalized, k, D)
    return distorted.reshape(-1, 2)
