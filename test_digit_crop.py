from functions.register_helmet import register_helmet
import cv2

test_image = cv2.imread("../pictures/number20.png")
register_helmet(test_image)