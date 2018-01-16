import cv2
import os
import numpy as np
from utils import normalize
from utils.edge_detection import edge_detect
from utils.edge_detection import marr_hildreth_method, canny_method, zero_crossing
from utils.threshold_segmentation import thresholds
from utils.threshold_segmentation import otsu_method


if __name__ == '__main__':

    fn = "./img/building.tif"
    if not os.path.exists(fn):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    n_img = normalize(img)
    ed_ro = edge_detect(n_img, mode='Roberts', dst="./img/edge/%s.tif")
    ed_pr = edge_detect(n_img, mode='Prewitt', dst="./img/edge/%s.tif")
    ed_so = edge_detect(n_img, mode='Sobel', dst="./img/edge/%s.tif")

    ed_mh = marr_hildreth_method(n_img, size_t=25, sigma=4., dst="./img/edge/%s.tif")
    ed_z1 = zero_crossing(np.copy(ed_mh), threshold=0, dst="./img/edge/%s-thr.0.tif")
    ed_z2 = zero_crossing(np.copy(ed_mh), threshold=0.04*np.max(ed_mh), dst="./img/edge/%s-thr.0.04max.tif")
    # print('---')
    # canny_method(img, size_t=25, sigma=4., tw=0.04, th=0.10, dst="./img/edge/%s.tif")

    print('...')
    print()

    fn = "./img/polymersomes.tif"
    if not os.path.exists(fn):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    ts = thresholds(img, dst="./img/segment/%s.tif")
    ot = otsu_method(img, dst="./img/segment/%s.tif")
