import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt


class Watermark:
    def __init__(self, image_path):
        self.source_image = cv.imread(image_path)
        self.watermark_image = cv.imread(image_path)
        pass

    def Encoding(self):
        print(self.source_image)
        pass

    def Decoding(self):
        pass


if __name__ == '__main__':
    image_path = '../Images/0001-055.TIF'
    Watermark(image_path).Encoding()
