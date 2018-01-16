import cv2
import numpy as np
import scipy as sp
import scipy.ndimage
from skimage.morphology import reconstruction


def normalize(img):

    _max = float(max(img.flatten()))
    _min = float(min(img.flatten()))
    w, h = img.shape
    for i in range(w):
        for j in range(h):
            img[i, j] = int(255 * (img[i, j] - _min) / _max)

    return img


def opening_reconstruct(img, dst=None):
    se0 = np.ones((51, 1), np.uint8)
    bw2 = cv2.erode(img, se0)
    bw3 = cv2.dilate(bw2, se0)
    cv2.imwrite(dst % 'marker', bw2)
    cv2.imwrite(dst % 'dilation', bw3)

    marker = np.copy(bw2)
    mask = np.copy(img)
    char = reconstruction(marker, mask)
    cv2.imwrite(dst % 'character', char)

    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(dst % 'built-in', result)


def filling(img, dst=None):

    bw2 = np.copy(img)
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    mask = sp.ndimage.binary_erosion(~np.isnan(bw2), structure=el)

    result = np.copy(bw2)
    result[mask] = 255
    bw3 = np.copy(bw2)
    bw3.fill(0)
    el = sp.ndimage.generate_binary_structure(2, 1).astype(np.int)

    while not np.array_equal(bw3, result):
        bw3 = np.copy(result)
        result = np.maximum(bw2, sp.ndimage.grey_erosion(result, size=(3,3), footprint=el))

    if dst:
        # cv2.imwrite(dst % 'mask', normalize(mask))
        cv2.imwrite(dst % 'template', (255 * np.ones_like(img, int) - img))
        cv2.imwrite(dst % 'built-in', result)

    return result


def border_clearing(img, dst=None):

    w, h = img.shape
    f = np.zeros((w, h), np.uint8)
    for i in range(w):
        f[i, 0] = img[i, 0]
        f[i, h-1] = img[i, h-1]
    for j in range(h):
        f[0, j] = img[0, j]
        f[w-1, j] = img[w-1, j]
    if dst:
        cv2.imwrite(dst % 'marker', f)
    se0 = np.ones((3, 3), np.uint8)
    hc = np.ones_like(img, int)
    for itr in range(20):
        a = cv2.dilate(f, se0)
        # if dst:
        #     cv2.imwrite(dst % ('itr' + str(itr)), a)
        hc = a & img
        f = hc
    h = img - hc

    if dst:
        cv2.imwrite(dst % 'final', h)

    return h


if __name__ == '__main__':
    pass
