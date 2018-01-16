import cv2
import os
import math
import numpy as np
from utils import parse_args, center_transform


def ideal_filter(fp, param, dst=None):

    if not os.path.exists(fp):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    w, h = img.shape

    img_dft = np.fft.fft2(center_transform(img))
    result = np.copy(img_dft)

    if param.mode == 'lo':
        for i in range(w):
            for j in range(h):
                if math.sqrt((i - w / 2) ** 2 + (j - h / 2) ** 2) > param.threshold:
                    result[i, j] = 0 + 0.j
    elif param.mode == 'hi':
        for i in range(w):
            for j in range(h):
                if math.sqrt((i - w / 2) ** 2 + (j - h / 2) ** 2) <= param.threshold:
                    result[i, j] = 0 + 0.j

    r_comp = np.real(np.fft.ifft2(result))
    img_ct = center_transform(r_comp, dst)

    return img_ct


if __name__ == '__main__':

    fn = "../img/characters_test_pattern.tif"
    opt = parse_args()
    opt.mode = 'hi'
    ideal_filter(fp=fn, param=opt, dst="../img/ideal_filter_hi.tif")
    opt.mode = 'lo'
    ideal_filter(fp=fn, param=opt, dst="../img/ideal_filter_lo.tif")
