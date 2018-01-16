import os
import cv2


def equalize(fp, dst):
    if not os.path.exists(fp):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img)
    cv2.imwrite(dst, equ)


if __name__ == '__main__':
    equalize("../img/Fig2.jpg", "../img/Fig2_hist.jpg")
    pass
