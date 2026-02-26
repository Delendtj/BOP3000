import cv2

def extract_helmet_box(det, frame , min_size=32):


    if det is None or len(det) == 0:
        return []

    # class filter, 0 = hjelmnummer, 1 = skøyter
    class_0_dets = det[:, 5] == 0
    class_0_valid_dets = det[class_0_dets]


    cropped_img = []
    for dets in class_0_valid_dets:
        x1, y1, x2, y2, conf, cls, track_id = dets

        #konverter til int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        # skip cropping og printing hvis detectionen er mindre enn min_size
        #if (x2 - x1) < min_size or y2 - y1 < min_size:
        #    continue


        cropped = frame[y1:y2, x1:x2].copy()

        cropped_img.append({
            'image': cropped,
            'bbox': (x1, y1, x2, y2),
            'conf': conf,
            'track_id': track_id,
        })

    return cropped_img