import cv2
import os
from utils.mask import zonal_mask, threshold_mask
from utils.compress import dwt

if __name__ == '__main__':

    fn = "./img/lenna.tif"
    if not os.path.exists(fn):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

    zonal_mask(img, dst="./img/dct/%s.tif")
    threshold_mask(img, dst="./img/dct/%s.tif")

    dwt(img, mode='haar', dst="./img/wavelets/%s.tif")
    dwt(img, mode='db4', dst="./img/wavelets/%s.tif")
    dwt(img, mode='sym4', dst="./img/wavelets/%s.tif")
    dwt(img, mode='bior6.8', dst="./img/wavelets/%s.tif")
    pass
