import cv2 as cv
import numpy as np
import random
import time


class Watermark:
    def __init__(self):
        self.watermark_image = cv.imread('../Watermark/logo.png')
        self.result_image_path = '../Watermark/result.jpg'
        self.random_seed = 2021
        self.alpha = 5

    def encoding(self, image_path):
        start_time = time.time()

        # Read Image
        source_image = cv.imread(image_path)

        source_height, source_width, _ = source_image.shape
        watermark_height, watermark_width, _ = self.watermark_image.shape

        print('source height : ', source_height, ', source_width : ', source_width)
        print('watermark height : ', watermark_height, ', watermark width : ', watermark_width)

        # Convert image to frequency area with Fast Fourier Transform (image -> frequency)
        source_frequency = np.fft.fft2(source_image)

        # Get random seed
        y_random_indices, x_random_indices = list(range(source_height)), list(range(source_width))
        random.seed(self.random_seed)
        random.shuffle(x_random_indices)
        random.shuffle(y_random_indices)

        print('y random seed : ', y_random_indices)
        print('x random seed : ', x_random_indices)

        # Injection watermark
        watermark_layer = np.zeros(source_image.shape, dtype=np.uint8)
        for y in range(watermark_height):
            for x in range(watermark_width):
                watermark_layer[y_random_indices[y], x_random_indices[x]] = self.watermark_image[y, x]

        # Encoding frequency area + watermark layer
        result_frequency = source_frequency + self.alpha * watermark_layer

        # Apply Inverse Fast Fourier Transform (frequency -> image)
        result_image = np.fft.ifft2(result_frequency)
        result_image = np.real(result_image)
        result_image = result_image.astype(np.uint8)

        # Show elapsed time
        end_time = time.time()
        print('Encoding elapsed time : ', end_time - start_time, '\n')

        # Visualization
        cv.imshow('source_image', source_image)
        cv.imshow('watermark', self.watermark_image)
        cv.imshow('watermark_layer', watermark_layer)
        cv.imshow('result_image', result_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

        return result_image

    def decoding(self, source_image_path, encoded_image):
        # Read Image
        source_image = cv.imread(source_image_path)

        source_height, source_width, _ = source_image.shape
        print('original_height : ', source_height)
        print('original_width : ', source_width)

        encoded_height, encoded_width, _ = encoded_image.shape

        # Convert image to frequency area with Fast Fourier Transform (image -> frequency)
        source_frequency = np.fft.fft2(source_image)
        encoded_frequency = np.fft.fft2(encoded_image)

        # Convert frequency area to image (frequency -> image)
        watermark_layer = (source_frequency - encoded_frequency) / self.alpha
        watermark_layer = np.real(watermark_layer).astype(np.uint8)

        # Get random seed
        y_random_indices, x_random_indices = list(range(encoded_height)), list(range(encoded_width))
        random.seed(self.random_seed)
        random.shuffle(x_random_indices)
        random.shuffle(y_random_indices)

        print('y random seed : ', y_random_indices)
        print('x random seed : ', x_random_indices)

        # Restore watermark
        result_image = np.zeros(watermark_layer.shape, dtype=np.uint8)

        for y in range(encoded_height):
            for x in range(encoded_width):
                result_image[y, x] = watermark_layer[y_random_indices[y], x_random_indices[x]]

        cv.imshow('original image', source_image)
        cv.imshow('target image', encoded_image)
        cv.imshow('watermark layer', watermark_layer)
        cv.imshow('result image', result_image)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    source_path = '../Watermark/jennie.jpg'

    protected_image = Watermark().encoding(source_path)
    Watermark().decoding(source_path, protected_image)
