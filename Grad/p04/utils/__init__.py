import cv2
import numpy as np
from optparse import OptionParser


def parse_args():

    parser = OptionParser()

    parser.add_option("-x", "--width", type="int", dest="width", default=448)
    parser.add_option("-y", "--height", type="int", dest="height", default=464)
    parser.add_option("-a", "--param-a", type="float", dest="a", default=0.0)
    parser.add_option("-b", "--param-b", type="float", dest="b", default=20.0)
    parser.add_option("-m", "--mode", type="string", dest="mode", default="uniform")

    options, arguments = parser.parse_args()
    return options


def get_noise(param):

    w, h = param.width, param.height
    a, b = param.a, param.b
    c = a + b
    if param.mode == "uniform":
        return a + (b - a) * np.random.rand(w, h)
    elif param.mode == "gaussian":
        return a + b * np.random.randn(w, h)
    elif param.mode == "salt":
        result = np.zeros((w, h), int)
        xm = np.random.rand(w, h)
        for i in range(w):
            for j in range(h):
                if a < xm[i, j] <= c:
                    result[i, j] = 255
        return result
    elif param.mode == "pepper":
        result = np.zeros((w, h), int)
        xm = np.random.rand(w, h)
        for i in range(w):
            for j in range(h):
                if xm[i, j] <= a:
                    result[i, j] = -255
        return result
    elif param.mode == "salt & pepper":
        result = np.zeros((w, h), int)
        xm = np.random.rand(w, h)
        for i in range(w):
            for j in range(h):
                if xm[i, j] <= a:
                    result[i, j] = -255
                elif a < xm[i, j] <= c:
                    result[i, j] = 255
        return result
    elif param.mode == "rayleigh":
        return a + (-b * np.log(-np.random.rand(w, h) + 1)) ** 0.5
    elif param.mode == "exponential":
        return (-1.0 / a) * np.log(-np.random.rand(w, h) + 1)
    elif param.mode == "erlang":
        result = np.zeros((w, h), int)
        for i in range(b):
            result = result + (-1.0/a) * np.log(-np.random.rand(w, h) + 1)
        return result

    return None


def truncate(img, dst=None):

    w, h = img.shape
    for i in range(w):
        for j in range(h):
            img[i, j] = min(max(img[i, j], 0), 255)

    if dst:
        cv2.imwrite(dst, img)

    return img


if __name__ == '__main__':
    pass
