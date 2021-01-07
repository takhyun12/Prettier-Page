import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt


class Watermark:
    def __init__(self, image_path):
        self.source_image = cv.imread(image_path)
        self.watermark_image = cv.imread('../Watermark/logo.png')

    def Encoding(self):
        source_height, source_width, _ = self.source_image.shape
        watermark_height, watermark_width, _ = self.watermark_image.shape

        cv.imshow('source_image', self.source_image)
        cv.imshow('watermark', self.watermark_image)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def Decoding(self):
        pass


if __name__ == '__main__':
    image_path = '../Watermark/jennie.jpg'
    Watermark(image_path).Encoding()
