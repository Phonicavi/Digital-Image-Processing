import cv2
import os
import numpy as np
from utils import normalize


def sobel_operation(fp, dst=None, opr_x=None, opr_y=None):

    if not os.path.exists(fp):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    w, h = img.shape

    uc = np.zeros((w + 2, h + 2), int)
    uc[1:-1, 1:-1] = img
    result = np.zeros_like(img, int)

    if not opr_x:  # default operator
        opr_x = [[-1, -2, -1],
                 [+0, +0, +0],
                 [+1, +2, +1]]

    if not opr_y:  # default operator
        opr_y = [[-1, +0, +1],
                 [-2, +0, +2],
                 [-1, +0, +1]]

    for i in range(w):
        for j in range(h):
            result[i, j] = abs(int(sum(sum(uc[i:i+3, j:j+3] * opr_x)))) + \
                           abs(int(sum(sum(uc[i:i+3, j:j+3] * opr_y))))

    if dst:
        cv2.imwrite(dst, normalize(result))

    return result


if __name__ == '__main__':

    sobel_operation(fp="../img/skeleton_orig.tif", dst="../img/sobel.tif")
