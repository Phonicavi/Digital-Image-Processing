import cv2
import math
import numpy as np
from utils import rmat


def translate(img, x, y, dst=None):

    w, h = img.shape
    result = np.zeros((w, h), int)
    for i in range(w):
        for j in range(h):
            if 0 <= i - x < w and 0 <= j - y < h:
                result[i, j] = img[i - x, j - y]

    if dst:
        cv2.imwrite(dst, result)

    return result


def rotate(img, angle, bbx='loose', mode='nn', dst=None):

    w, h = img.shape
    center = np.array([int(w/2), int(h/2)])

    rad = math.pi * (angle / 180.0)
    rmt1 = rmat(rad)
    rmt2 = rmat(-rad)

    if bbx == 'loose':
        v1 = np.array([0, 0])           # left-top
        v2 = np.array([w - 1, 0])       # right-top
        v3 = np.array([0, h - 1])       # left-bottom
        v4 = np.array([w - 1, h - 1])   # right-bottom
        u1 = np.dot((v1 - center), rmt1) + center
        u2 = np.dot((v2 - center), rmt1) + center
        u3 = np.dot((v3 - center), rmt1) + center
        u4 = np.dot((v4 - center), rmt1) + center
        w2s = [u1[0], u2[0], u3[0], u4[0]]
        h2s = [u1[1], u2[1], u3[1], u4[1]]
        border = [int(min(w2s)), int(max(w2s)), int(min(h2s)), int(max(h2s))]
    elif bbx == 'crop':
        border = np.array([0, w, 0, h])
    else:
        border = np.array([0, w, 0, h])
    result = np.zeros((border[1] - border[0], border[3] - border[2]), int)

    if mode == 'nn':    # nearest neighbour
        for i in range(border[0], border[1], 1):
            for j in range(border[2], border[3], 1):
                tp = (np.dot((np.array([i, j]) - center), rmt2) + center).astype(int)
                tx, ty = tp[0], tp[1]
                if 0 <= tx < w and 0 <= ty < h:
                    result[i - border[0], j - border[2]] = img[tx, ty]
    elif mode == 'bi':  # bilinear interpolation
        for i in range(border[0], border[1], 1):
            for j in range(border[2], border[3], 1):
                tr = np.dot((np.array([i, j]) - center), rmt2) + center
                tp = tr.astype(int)
                tx, ty = tp[0], tp[1]
                if 0 <= tx < w and 0 <= ty < h:
                    x1 = int(tx - 0.5)
                    x2 = int(tx + 0.5)
                    y1 = int(ty - 0.5)
                    y2 = int(ty + 0.5)
                    if x1 < 0 or x2 >= w or y1 < 0 or y2 >= h:
                        result[i - border[0], j - border[2]] = img[tx, ty]
                    else:
                        A = np.matrix([
                            [x1, y1, x1*y1, 1],
                            [x1, y2, x1*y2, 1],
                            [x2, y1, x2*y1, 1],
                            [x2, y2, x2*y2, 1],
                        ]).astype(float)
                        curr_p_a = np.array([tr[0], tr[1], tr[0]*tr[1], 1]).astype(float)
                        B = np.matrix([
                            [img[y1, x1]],
                            [img[y2, x1]],
                            [img[y1, x2]],
                            [img[y2, x2]],
                        ]).astype(float)
                        K = np.linalg.lstsq(A, B)[0]
                        result[i - border[0], j - border[2]] = np.dot(curr_p_a, K)[0, 0]

    if dst:
        cv2.imwrite(dst, result)

    return result


def scale(img, px, py, mode='nn', dst=None):

    w, h = img.shape
    w2, h2 = int(w * px), int(h * py)
    imat = np.array([[1./px, 0.], [0., 1./py]])
    result = np.zeros((h2, w2), int)

    if mode == 'nn':
        for i in range(w2):
            for j in range(h2):
                tp = (np.dot(imat, np.matrix([[i], [j]]).astype(float))).astype(int)
                ty, tx = tp[0], tp[1]
                if 0 <= tx < w and 0 <= ty < h:
                    result[j, i] = img[tx, ty]
    elif mode == 'bi':
        for i in range(w2):
            for j in range(h2):
                tr = np.dot(imat, np.matrix([[i], [j]]).astype(float))
                tp = tr.astype(int)
                tx, ty = tp[0], tp[1]
                if 0 <= tx < w and 0 <= ty < h:
                    x1 = int(tx - 0.5)
                    x2 = int(tx + 0.5)
                    y1 = int(ty - 0.5)
                    y2 = int(ty + 0.5)
                    if x1 < 0 or x2 >= w or y1 < 0 or y2 >= h:
                        result[j, i] = img[tx, ty]
                    else:
                        A = np.matrix([
                            [x1, y1, x1 * y1, 1],
                            [x1, y2, x1 * y2, 1],
                            [x2, y1, x2 * y1, 1],
                            [x2, y2, x2 * y2, 1],
                        ]).astype(float)
                        curr_p_a = np.array([tr[0], tr[1], tr[0] * tr[1], 1]).astype(float)
                        B = np.matrix([
                            [img[y1, x1]],
                            [img[y2, x1]],
                            [img[y1, x2]],
                            [img[y2, x2]],
                        ]).astype(float)
                        K = np.linalg.lstsq(A, B)[0]
                        result[j, i] = np.dot(curr_p_a, K)[0, 0]
                        pass

    if dst:
        cv2.imwrite(dst, result)

    return result


if __name__ == '__main__':
    pass
