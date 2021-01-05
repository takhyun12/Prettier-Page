import cv2 as cv
import os
import imutils
import numpy as np


def Image_Process(image_path):
    # Image Read
    source_image = cv.imread(image_path)
    ratio = source_image.shape[0] / 500.0
    copied_image = source_image.copy()
    copied_image = imutils.resize(copied_image, height=500)

    # Apply GaussianBlur + OTSU-Thresholding
    grayscale_image = cv.cvtColor(copied_image, cv.COLOR_BGR2GRAY)
    grayscale_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)
    ret, grayscale_image = cv.threshold(grayscale_image, 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # cv.imshow("grayscale_image", grayscale_image)

    # Find Contours
    contours, hierarchy = cv.findContours(grayscale_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    contour_image = copied_image.copy()
    cv.drawContours(contour_image, [biggest_contour], 0, (0, 0, 255), 2)
    cv.imshow('contour_image', contour_image)

    # Crop Image
    x, y, w, h = cv.boundingRect(biggest_contour)
    x = int(x * ratio)
    y = int(y * ratio)
    w = int(w * ratio)
    h = int(h * ratio)

    cropped_image = source_image[y:y + h, x:x + w]

    cv.imshow(image_path, cropped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save Image
    # directory_name = '../Result'
    # image_name = os.path.basename(image_path)
    # result_image_path = os.path.join(directory_name, image_name)
    # cv.imwrite(result_image_path, cropped_image)
    # print('[>] ' + result_image_path)


class PrettierPage:
    def __init__(self):
        pass

    @staticmethod
    def Process_File(image_path):
        Image_Process(image_path)

    @staticmethod
    def Process_Directory(directory_path):
        image_list = os.listdir(directory_path)
        for image_name in image_list:
            image_path = os.path.join(directory_path, image_name)
            Image_Process(image_path)


if __name__ == '__main__':
    # PrettierPage.Process_File('../Images/0001-001.TIF')
    PrettierPage.Process_Directory('../Images')