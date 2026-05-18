import cv2

cap = cv2.VideoCapture(2)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame = cap.read()
print(frame)