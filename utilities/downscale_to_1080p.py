import cv2

def downscale_to_1080p(frame):
    """
    Downscales input frame to 1080p if the image is non 1080p
    """

    if frame is None: return frame

    h, w = frame.shape[:2]
    if w != 1920 or h != 1080:
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    return frame