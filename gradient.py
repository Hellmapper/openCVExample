import cv2
import numpy
import numpy as np

img = cv2.imread("lena.jpg", cv2.IMREAD_COLOR)
img = cv2.GaussianBlur(img, (51,51), 12220)
cv2.imshow('source', img)
cv2.waitKey()