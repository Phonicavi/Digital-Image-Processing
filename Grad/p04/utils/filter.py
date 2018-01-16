import cv2
import os
import numpy as np


def arithmetic_mean_filter(img, size, param=None):

    w, h = img.shape
    x, y = size
    margin_x = int(x / 2.0)
    margin_y = int(y / 2.0)
    uc = np.zeros((w + 2 * margin_x, h + 2 * margin_y), int)

    uc[margin_x:-margin_x, margin_y:-margin_y] = img
    for i in range(w):
        for j in range(h):
            img[i, j] = int(np.mean(uc[i:i + x, j:j + y]))

    return img


def geometric_mean_filter(img, size, param=None):

    w, h = img.shape
    x, y = size
    margin_x = int(x / 2.0)
    margin_y = int(y / 2.0)
    uc = np.ones((w + 2 * margin_x, h + 2 * margin_y), int)
    uc[margin_x:-margin_x, margin_y:-margin_y] = img

    for i in range(w):
        for j in range(h):
            img[i, j] = int(np.prod(uc[i:i + x, j:j + y] ** (1.0 / (x * y))))

    return img


def contraharmonic_mean_filter(img, size, param=1):

    w, h = img.shape
    x, y = size
    margin_x = int(x / 2.0)
    margin_y = int(y / 2.0)
    uc = np.zeros((w + 2 * margin_x, h + 2 * margin_y), int)

    uc[margin_x:-margin_x, margin_y:-margin_y] = img
    for i in range(w):
        for j in range(h):
            try:
                t1 = uc[i:i + x, j:j + y] ** (param + 1)
                t2 = uc[i:i + x, j:j + y] ** param
                t1[t1 == np.inf] = 0.0
                t2[t2 == np.inf] = 0.0
                t1[t1 == np.NINF] = 0.0
                t2[t2 == np.NINF] = 0.0
                img[i, j] = int(sum(sum(t1)) / sum(sum(t2)))
            except Exception as e:
                img[i, j] = 0
                # print i, j
                # print t2

    return img


def median_filter(img, size, param=None):

    w, h = img.shape
    x, y = size
    margin_x = int(x / 2.0)
    margin_y = int(y / 2.0)
    uc = np.zeros((w + 2 * margin_x, h + 2 * margin_y), int)

    uc[margin_x:-margin_x, margin_y:-margin_y] = img
    for i in range(w):
        for j in range(h):
            img[i, j] = int(np.median(uc[i:i + x, j:j + y]))

    return img


def max_filter(img, size, param=None):

    w, h = img.shape
    x, y = size
    margin_x = int(x / 2.0)
    margin_y = int(y / 2.0)
    uc = np.zeros((w + 2 * margin_x, h + 2 * margin_y), int)

    uc[margin_x:-margin_x, margin_y:-margin_y] = img
    for i in range(w):
        for j in range(h):
            img[i, j] = int(np.max(uc[i:i + x, j:j + y]))

    return img


def min_filter(img, size, param=None):

    w, h = img.shape
    x, y = size
    margin_x = int(x / 2.0)
    margin_y = int(y / 2.0)
    uc = np.zeros((w + 2 * margin_x, h + 2 * margin_y), int)

    uc[margin_x:-margin_x, margin_y:-margin_y] = img
    for i in range(w):
        for j in range(h):
            img[i, j] = int(np.min(uc[i:i + x, j:j + y]))

    return img


def alpha_mean_filter(img, size, param=5):

    w, h = img.shape
    x, y = size
    margin_x = int(x / 2.0)
    margin_y = int(y / 2.0)
    uc = np.zeros((w + 2 * margin_x, h + 2 * margin_y), int)

    uc[margin_x:-margin_x, margin_y:-margin_y] = img
    for i in range(w):
        for j in range(h):
            core = uc[i:i + x, j:j + y].flatten()
            img[i, j] = int(np.mean(core[param/2:-param/2]))

    return img


if __name__ == '__main__':
    pass
