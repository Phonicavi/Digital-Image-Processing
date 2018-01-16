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


def gaussian_fspecial(shape=(3,3),sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp( -(x * x + y * y) / (2. * sigma * sigma) )
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    return h


if __name__ == '__main__':
    pass
