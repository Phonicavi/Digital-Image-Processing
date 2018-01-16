import cv2
import numpy as np
from utils import coefficient, restore_coefficient


size_t = 8

t0 = np.array([
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

a1 = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68,  109, 103, 77],
    [24, 35, 55, 64, 81,  104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def zonal_mask(img, dst=None):
    w, h = img.shape
    uc = np.zeros((w, h), np.uint8)
    for i in range(w / size_t):
        for j in range(h / size_t):
            sub_img = img[i*size_t:(i+1)*size_t, j*size_t:(j+1)*size_t]
            co = coefficient(sub_img, t0, 0, size_t)
            co = restore_coefficient(co, size_t)
            uc[i * size_t:(i + 1) * size_t, j * size_t:(j + 1) * size_t] = co
    diff = img - uc
    if dst:
        cv2.imwrite(dst % 'zonal_mask', uc)
        cv2.imwrite(dst % 'zonal_diff', diff)
    return uc, diff


def threshold_mask(img, dst=None):
    w, h = img.shape
    uc = np.zeros((w, h), np.uint8)
    for i in range(w / size_t):
        for j in range(h / size_t):
            sub_img = img[i * size_t:(i + 1) * size_t, j * size_t:(j + 1) * size_t]
            co = coefficient(sub_img, sub_img / a1, 1.0/2/255, size_t)
            co = restore_coefficient(co, size_t)
            uc[i * size_t:(i + 1) * size_t, j * size_t:(j + 1) * size_t] = co
    diff = img - uc
    if dst:
        cv2.imwrite(dst % 'threshold_mask', uc)
        cv2.imwrite(dst % 'threshold_diff', diff)
    return uc, diff


if __name__ == '__main__':

    pass
