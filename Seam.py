import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def Encoding():
    img = cv2.imread('../Watermark/01.jpg')
    img_wm = cv2.imread('../Watermark/logo2.png')

    height, width, _ = img.shape
    wm_height, wm_width, _ = img_wm.shape

    img_f = np.fft.fft2(img)

    y_random_indices, x_random_indices = list(range(height)), list(range(width))
    random.seed(2021)
    random.shuffle(x_random_indices)
    random.shuffle(y_random_indices)

    random_wm = np.zeros(img.shape, dtype=np.uint8)

    for y in range(wm_height):
        for x in range(wm_width):
            random_wm[y_random_indices[y], x_random_indices[x]] = img_wm[y, x]

    plt.figure(figsize=(16, 10))

    alpha = 5

    result_f = img_f + alpha * random_wm

    result = np.fft.ifft2(result_f)
    result = np.real(result)
    result = result.astype(np.uint8)

    cv2.imshow('result', result)
    cv2.imwrite('../Watermark/result.jpg', result)

    img = cv2.imread('../Watermark/01.jpg')
    # result = cv2.imread('../Watermark/result.jpg')

    height, width, _ = img.shape

    img_ori_f = np.fft.fft2(img)
    img_input_f = np.fft.fft2(result)

    alpha = 5
    watermark = (img_ori_f - img_input_f) / alpha
    watermark = np.real(watermark).astype(np.uint8)

    plt.figure(figsize=(16, 10))
    plt.imshow(watermark)

    y_random_indices, x_random_indices = list(range(height)), list(range(width))
    random.seed(2021)
    random.shuffle(x_random_indices)
    random.shuffle(y_random_indices)

    result2 = np.zeros(watermark.shape, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            result2[y, x] = watermark[y_random_indices[y], x_random_indices[x]]

    cv2.imshow('result2', result2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

Encoding()