import cv2
import numpy as np

# import ideal_filter as idf
# import butterworth_filter as bwf
# import gaussian_filter as gsf

from optparse import OptionParser


def parse_args():

    parser = OptionParser()

    parser.add_option("-d", "--threshold", type="float", dest="threshold", default=30.0)
    parser.add_option("-n", "--dimension", type="float", dest="dim", default=2)
    parser.add_option("-m", "--mode", type="string", dest="mode", default="hi")

    options, arguments = parser.parse_args()
    return options


def normalize(img):

    _max = float(max(img.flatten()))
    _min = float(min(img.flatten()))
    w, h = img.shape
    for i in range(w):
        for j in range(h):
            img[i, j] = int(255 * (img[i, j] - _min) / _max)

    return img


def center_transform(img, dst=None):

    w, h = img.shape
    ic = np.zeros((w, h), int)
    for i in range(w):
        for j in range(h):
            ic[i, j] = ((-1)**(i + j)) * img[i, j]

    result = normalize(ic)
    if dst:
        cv2.imwrite(dst, result)

    return ic


if __name__ == '__main__':

    fn = "../img/characters_test_pattern.tif"
    im = cv2.cvtColor(cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    center_transform(im, dst="../img/center_transformed.tif")
