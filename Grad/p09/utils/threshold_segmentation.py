import cv2
import numpy as np


def thresholds(img, dst):
    w, h = img.shape
    result = np.zeros((w, h), np.uint8)
    t1 = np.mean(img)
    t2 = -1
    t0 = 10 ** -4
    while abs(t1 - t2) > t0:
        t2 = t1
        g1 = img[np.where(img > t1)]
        g2 = img[np.where(img <= t1)]
        m1 = np.mean(g1)
        m2 = np.mean(g2)
        t1 = (m1 + m2) / 2.0
    result[np.where(img > t1)] = 255
    result[np.where(img <= t1)] = 0
    if dst:
        cv2.imwrite(dst % 'poly-ts', result)
    return result


def otsu_method(img, dst=None):
    w, h = img.shape
    result = np.zeros_like(img, np.uint8)
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    hist = hist.astype(float) / (w * h)
    color = np.array([c * hist[c] for c in range(256)])
    accum = np.array([np.sum(hist[:i+1]) for i in range(256)])
    mtab = np.array([np.sum(color[:i+1]) for i in range(256)])
    mg = mtab[-1]
    B = (mg * accum - mtab) ** 2 / (accum * (1 - accum))
    B[np.isnan(B)] = 0.0
    B[B <= (10 ** -8)] = 0.0
    th = np.round(np.mean(np.where(B == np.max(B))))
    result[np.where(img > th)] = 255
    result[np.where(img <= th)] = 0
    if dst:
        cv2.imwrite(dst % 'otsu_result', result)
    return result


if __name__ == '__main__':
    pass
