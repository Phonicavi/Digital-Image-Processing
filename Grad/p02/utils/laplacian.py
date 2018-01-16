import cv2
import os
import numpy as np
from utils import normalize


def laplace_operation(fp, dst=None, opr=None):

    if not os.path.exists(fp):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    w, h = img.shape

    uc = np.zeros((w + 2, h + 2), int)
    uc[1:-1, 1:-1] = img
    result = np.zeros_like(img, int)

    if not opr:  # default operator
        opr = [[-1, -1, -1],
               [-1, +8, -1],
               [-1, -1, -1]]

    for i in range(w):
        for j in range(h):
            result[i, j] = int(sum(sum(uc[i:i+3, j:j+3] * opr)))

    if dst:
        cv2.imwrite(dst, normalize(result))

    return result


if __name__ == '__main__':

    laplace_operation(fp="../img/skeleton_orig.tif", dst="../img/laplace.tif")
