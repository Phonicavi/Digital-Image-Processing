import cv2
import os
import numpy as np
from matplotlib import pyplot as plt


def histogram(fp, lvs):

    if not os.path.exists(fp):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    plt.hist(img.flatten(), lvs, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.show()


def equalize(fp, dst):

    if not os.path.exists(fp):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])

    pix_sum = sum(hist)
    equ = np.zeros_like(img)
    w, h = equ.shape
    for i in range(w):
        for j in range(h):
            gr = int(img[i][j])
            equ[i][j] = 255 - int(255.0 * sum(hist[gr+1:]) / pix_sum)

    cv2.imwrite(dst, equ)


if __name__ == '__main__':
    equalize("../img/Fig1.jpg", "../img/Fig1_hist.jpg")
    histogram("../img/Fig1.jpg" , 30)
    histogram("../img/Fig1_hist_eq.jpg", 30)
    histogram("../img/Fig1_hist.jpg", 30)
    pass
