import cv2 as cv
import numpy as np
import random


class Watermark:
    def __init__(self, image_path):
        self.source_image = cv.imread(image_path)
        self.watermark_image = cv.imread('../Watermark/logo.png')
        self.random_seed = 2021
        self.alpha = 5

    def Encoding(self):
        source_height, source_width, _ = self.source_image.shape
        watermark_height, watermark_width, _ = self.watermark_image.shape

        # Convert image to frequency area with Fast Fourier Transform (image -> frequency)
        source_frequency = np.fft.fft2(self.source_image)

        # Get random seed
        y_random_indices, x_random_indices = list(range(source_height)), list(range(source_width))
        random.seed(self.random_seed)
        random.shuffle(x_random_indices)
        random.shuffle(y_random_indices)

        print('y random seed', y_random_indices)
        print('x random seed', x_random_indices)

        # Injection watermark
        watermark_layer = np.zeros(self.source_image.shape, dtype=np.uint8)
        for y in range(watermark_height):
            for x in range(watermark_width):
                watermark_layer[y_random_indices[y], x_random_indices[x]] = self.watermark_image[y, x]

        # Encoding frequency area + watermark layer
        result_frequency = source_frequency + self.alpha * watermark_layer

        # apply Inverse Fast Fourier Transform (frequency -> image)
        result_image = np.fft.ifft2(result_frequency)
        result_image = np.real(result_image)
        result_image = result_image.astype(np.uint8)

        # Visualization
        cv.imshow('source_image', self.source_image)
        cv.imshow('watermark', self.watermark_image)
        cv.imshow('watermark_layer', watermark_layer)
        cv.imshow('result_image', result_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def Decoding(self):
        pass


if __name__ == '__main__':
    image_path = '../Watermark/jennie.jpg'
    Watermark(image_path).Encoding()
