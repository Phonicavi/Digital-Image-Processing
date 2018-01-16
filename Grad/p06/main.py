import cv2
import os
from utils.transform import translate, rotate, scale


if __name__ == '__main__':

    fn = "./img/ray_trace_bottle.tif"
    if not os.path.exists(fn):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

    translate(img, x=30, y=50, dst="./img/ray_translated[30,50].tif")
    translate(img, x=-40, y=20, dst="./img/ray_translated[-40,20].tif")
    rotate(img, 30, bbx='loose', mode='nn', dst="./img/ray_rotate[30,l,nn].tif")
    rotate(img, 30, bbx='crop', mode='nn', dst="./img/ray_rotate[30,c,nn].tif")
    rotate(img, 30, bbx='loose', mode='bi', dst="./img/ray_rotate[30,l,bi].tif")
    rotate(img, 30, bbx='crop', mode='bi', dst="./img/ray_rotate[30,c,bi].tif")
    scale(img, px=1.5, py=2, mode='nn', dst="./img/ray_scale[1.5,2,nn].tif")
    scale(img, px=1.5, py=2, mode='bi', dst="./img/ray_scale[1.5,2,bi].tif")
