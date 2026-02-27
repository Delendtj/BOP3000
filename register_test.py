from functions.register_helmet import register_helmet
import cv2 as cv

img = cv.imread("../pictures/number204.png")
register_helmet(img, debug=True)
