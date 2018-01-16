import cv2
import os
import numpy as np


def normalize(img):

    _max = float(max(img.flatten()))
    _min = float(min(img.flatten()))
    w, h = img.shape
    for i in range(w):
        for j in range(h):
            img[i, j] = int(255 * (img[i, j] - _min) / _max)

    return img


def sharpen(fp, dst, enhancement):

    if not os.path.exists(fp):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    if img.shape != enhancement.shape:
        raise Exception('[Error] incompatible image size')

    cv2.imwrite(dst, img + enhancement)

    return img + enhancement


def mean_filter(img, size=5):

    w, h = img.shape
    margin = int(size/2.0)
    uc = np.zeros((w + 2 * margin, h + 2 * margin), int)
    uc[margin:-margin, margin:-margin] = img

    unit = [[1.0/size**2] * size] * size
    for i in range(w):
        for j in range(h):
            img[i, j] = int(sum(sum(uc[i:i+size, j:j+size] * unit)))

    return img


if __name__ == '__main__':

    filter1 = [[+0, -1, +0],
               [-1, +5, -1],
               [+0, -1, +0]]

    filter2 = [[-1, -1, -1],
               [-1, +8, -1],
               [-1, -1, -1]]

    # laps = laplace_operation(fp="../img/skeleton_orig.tif", opr=filter2)
    # sharpen(fp="../img/skeleton_orig.tif", dst="../img/sharpen2.tif", enhancement=laps)
    pass