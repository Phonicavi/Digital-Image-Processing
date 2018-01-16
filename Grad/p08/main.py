import cv2
import os
from utils import opening_reconstruct, filling, border_clearing


if __name__ == '__main__':

    fn = "./img/text_image.tif"
    if not os.path.exists(fn):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

    opening_reconstruct(img, dst="./img/opening/opening_%s.tif")
    filling(img, dst="./img/filling/filling_%s.tif")
    border_clearing(img, dst="./img/clearing/clearing_%s.tif")
