import cv2
import numpy as np


def normalize(img):

    _max = float(max(img.flatten()))
    _min = float(min(img.flatten()))
    w, h = img.shape
    for i in range(w):
        for j in range(h):
            img[i, j] = int(255 * (img[i, j] - _min) / _max)

    return img


def center_transform(img, dst=None):

    w, h = img.shape
    ic = np.zeros((w, h), int)
    for i in range(w):
        for j in range(h):
            ic[i, j] = ((-1)**(i + j)) * img[i, j]

    result = normalize(ic)
    if dst:
        cv2.imwrite(dst, result)

    return ic


def truncate(img, dst=None):

    w, h = img.shape
    for i in range(w):
        for j in range(h):
            img[i, j] = min(max(img[i, j], 0), 255)

    if dst:
        cv2.imwrite(dst, img)

    return img


def get_gaussian_noise(img, a, b):

    w, h = img.shape
    return a + b * np.random.randn(w, h)


if __name__ == '__main__':
    pass
