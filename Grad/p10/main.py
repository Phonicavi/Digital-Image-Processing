import cv2
import os
from utils.bound_follow import boundary
from utils.image_descript import description


if __name__ == '__main__':

    fn = "./img/noisy_stroke.tif"
    if not os.path.exists(fn):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    boundary(img, dst="./img/boundary/%s.tif")

    folder = "./img/washingtonDC/"
    img_list = []
    for fn in os.listdir(folder):
        fp = os.path.join(folder, fn)
        if os.path.isfile(fp):
            img_list.append(cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY))
    description(img_list, dst="./img/washingtonDC/%s.tif")
