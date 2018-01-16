import cv2
import math
import numpy as np


def transform(img, u, v, size, mode='dct'):
    ans = 0.0
    if mode == 'dct':
        for i in range(size):
            for j in range(size):
                tmp = 1.0
                tmp *= math.sqrt(2.0 / size) if i else math.sqrt(1.0 / size)
                tmp *= math.sqrt(2.0 / size) if j else math.sqrt(1.0 / size)
                ans += img[i, j] * tmp * \
                       math.cos((2 * u + 1) * i * math.pi / 2.0 / size) * \
                       math.cos((2 * v + 1) * j * math.pi / 2.0 / size)
    elif mode == 'idct':
        for i in range(size):
            for j in range(size):
                tmp = 1.0
                tmp *= math.sqrt(2.0 / size) if u else math.sqrt(1.0 / size)
                tmp *= math.sqrt(2.0 / size) if v else math.sqrt(1.0 / size)
                ans += img[i, j] * tmp * \
                       math.cos((2 * i + 1) * u * math.pi / 2.0 / size) * \
                       math.cos((2 * j + 1) * v * math.pi / 2.0 / size)
    return ans


def coefficient(img, mask, threshold, size):
    w, h = img.shape
    co = np.zeros((w, h), float)
    for i in range(w):
        for j in range(h):
            if mask[i, j] > threshold:
                co[i, j] = transform(img, i, j, size, mode='dct')
            else:
                co[i, j] = 0.0
    return co


def restore_coefficient(img, size):
    w, h = img.shape
    im = np.zeros((w, h), float)
    for i in range(w):
        for j in range(h):
            im[i, j] = transform(img, i, j, size, 'idct')
    return im


if __name__ == '__main__':
    pass
