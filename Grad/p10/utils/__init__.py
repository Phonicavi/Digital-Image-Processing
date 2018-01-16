import cv2
import math
import numpy as np


def mean_filter(img, size_t=5, dst=None):
    w, h = img.shape
    margin = int(size_t) / 2
    result = np.zeros((w, h), np.uint8)
    for i in range(w):
        for j in range(h):
            if (margin <= i < w - margin) and (margin <= j < h - margin):
                unit = img[i-margin:i-margin+size_t, j-margin:j-margin+size_t]
                result[i, j] = np.mean(unit)
    if dst:
        cv2.imwrite(dst % ('noisy_mean[' + str(size_t) + ']'), result)
    return result


def binary_filter(img, percent=0.90, dst=None):
    result = np.zeros_like(img, np.uint8)
    result[np.where(img >= 255 * percent)] = 255
    result[np.where(img < 255 * percent)] = 0
    if dst:
        cv2.imwrite(dst % ('noisy_binary[' + str(percent) + ']'), result)
    return result


def start_point(img):
    w, h = img.shape
    val = w + h
    a, b = w, h
    for i in range(w):
        for j in range(h):
            if (img[i, j] == 255) and (i + j <= val):
                a, b = i, j
                val = i + j
    return a, b


def rotate_from(p1, p2):
    p0 = np.array([p2[0] - p2[1] + p1[1], p2[1] + p2[0] - p1[0]])
    if p1[0] + p1[1] == p2[0] + p2[1]:
        p0[0] += p1[0] - p2[0]
    if p1[1] - p1[0] == p2[1] - p2[0]:
        p0[1] += p1[0] - p2[0]
    return p0


if __name__ == '__main__':
    pass
