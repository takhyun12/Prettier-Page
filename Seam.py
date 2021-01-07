import cv2 as cv
import os
import imutils
import numpy as np

watermark_image = cv.imread('../Watermark/jennie.jpg')

cv.imshow('watermark', watermark_image)
cv.waitKey(0)
cv.destroyAllWindows()