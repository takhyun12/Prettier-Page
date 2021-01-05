import cv2 as cv
import os
import imutils
import numpy as np


# Image Resizing
source_image = cv.imread('tt')
ratio = source_image.shape[0] / 500.0
copied_image = source_image.copy()
copied_image = imutils.resize(copied_image, height=500)
cv.imshow("copied_image", copied_image)

# Draw Contours
cv.drawContours(copied_image, ['biggest_contour'], 0, (0, 0, 255), 2)
cv.imshow('tt', copied_image)