from typing import List
import cv2 as cv
import os
import imutils
import numpy as np
import time

def findContoursInDark(copied_image):
    # Apply GaussianBlur + OTSU-Thresholding
    grayscale_image = cv.cvtColor(copied_image, cv.COLOR_BGR2GRAY)
    grayscale_image = cv.GaussianBlur(grayscale_image, (5, 5), 0)
    ret, grayscale_image = cv.threshold(grayscale_image, 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Apply Morph erode and open
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    erode_image = cv.erode(grayscale_image, kernel, iterations=1)
    morph_opened_image = cv.morphologyEx(erode_image, cv.MORPH_OPEN, kernel)

    # Find Contours
    contours, hierarchy = cv.findContours(morph_opened_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda value: value[0])[1]

    '''
    contour_image = copied_image.copy()
    cv.drawContours(contour_image, [biggest_contour], 0, (0, 0, 255), 2)
    cv.imshow('contour_image', contour_image)
    '''

    return biggest_contour


class PrettierPage:
    def __init__(self, image_path):
        self.source_image = cv.imread(image_path)  # type: cv
        self.source_ratio = self.source_image.shape[0] / 500.0
        self.image_path = image_path

        directory_name = '../Result'
        image_name = os.path.basename(image_path)
        self.result_image_path = os.path.join(directory_name, image_name)

    def makePretty(self):
        # Image Resizing
        copied_image = self.source_image.copy()
        copied_image = imutils.resize(copied_image, height=500)

        # Find Contours
        biggest_contour = findContoursInDark(copied_image)

        # Crop Image with source_image
        x, y, w, h = map(int, cv.boundingRect(biggest_contour) * np.array([self.source_ratio, self.source_ratio,
                                                                           self.source_ratio, self.source_ratio]))
        result_image = self.source_image[y: y + h, x: x + w]

        # Save Image
        cv.imwrite(self.result_image_path, result_image)
        print('[>] ' + self.result_image_path)


def Make_Pretty_With_Directory(directory_path):
    allow_extension_list = [".TIF", ".GIF", ".JPG", ".JPEG"]  # type: List[str]

    image_list = os.listdir(directory_path)
    for image_name in image_list:
        # Parse image path and extension
        image_path = os.path.join(directory_path, image_name)
        image_extension = os.path.splitext(image_name)[1].upper()  # type: str

        # Check image extension
        if image_extension not in allow_extension_list:
            return

        if os.path.isfile(image_path):
            PrettierPage(image_path).makePretty()


if __name__ == '__main__':
    # PrettierPage('../Images/0067.TIF').makePretty()
    Make_Pretty_With_Directory('../Images')

