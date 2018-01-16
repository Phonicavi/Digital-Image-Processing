import cv2
import math
import numpy as np
# import scipy.ndimage as spi
from utils import normalize, gaussian_fspecial


dh_core = np.array([[+1, +1, +1],
                    [+1, -8, +1],
                    [+1, +1, +1]])
dx_core = np.array([[+0, +0, +0],
                    [+0, -1, +1],
                    [+0, -1, +1]])
dy_core = np.array([[+0, +0, +0],
                    [+0, +1, +1],
                    [+0, -1, -1]])


def edge_detect(img, mode='Roberts', dst=None):
    w, h = img.shape
    result = np.copy(img)
    if mode == 'Roberts':
        opr_x = [[+0, +0, +0],
                 [+0, -1, +0],
                 [+0, +0, +1]]
        opr_y = [[+0, +0, +0],
                 [+0, +0, -1],
                 [+0, +1, +0]]
    elif mode == 'Prewitt':
        opr_x = [[-1, -1, -1],
                 [+0, +0, +0],
                 [+1, +1, +1]]
        opr_y = [[-1, +0, +1],
                 [-1, +0, +1],
                 [-1, +0, +1]]
    elif mode == 'Sobel':
        opr_x = [[-1, -2, -1],
                 [+0, +0, +0],
                 [+1, +2, +1]]
        opr_y = [[-1, +0, +1],
                 [-2, +0, +2],
                 [-1, +0, +1]]
    else:
        raise Exception('__unknown_mode__')
    uc = np.zeros((w + 2, h + 2), np.uint8)
    uc[1:-1, 1:-1] = img
    for i in range(w):
        for j in range(h):
            unit = uc[i:i+3, j:j+3]
            result[i, j] = abs(int(sum(sum(unit * opr_x)))) + abs(int(sum(sum(unit * opr_y))))
    if dst:
        cv2.imwrite(dst % (mode + '_result'), result)
    return


def marr_hildreth_method(img, size_t, sigma, dst=None):
    gm = gaussian_fspecial(shape=(size_t, size_t), sigma=sigma)
    # fm = spi.convolve(img, gm, mode='nearest')
    # cm = spi.convolve(fm, dh_core, mode='nearest')
    fm = cv2.filter2D(src=img, ddepth=-1, kernel=gm)
    cm = cv2.filter2D(src=fm, ddepth=-1, kernel=dh_core)
    if dst:
        cv2.imwrite(dst % 'Marr-Hildreth_result', normalize(np.copy(cm)))
    return cm


def canny_method(img, size_t, sigma, tw, th, dst=None):
    w, h = img.shape
    gm = gaussian_fspecial(shape=(size_t, size_t), sigma=sigma)
    # fm = spi.correlate(img, gm, mode='nearest')
    # gx = spi.correlate(fm, dx_core, mode='nearest')
    # gy = spi.correlate(fm, dy_core, mode='nearest')
    fm = cv2.filter2D(src=img, ddepth=-1, kernel=gm)
    gx = cv2.filter2D(src=fm, ddepth=-1, kernel=dx_core)
    gy = cv2.filter2D(src=fm, ddepth=-1, kernel=dy_core)
    vm = (gx ** 2 + gy ** 2) ** 0.5
    alpha = np.tanh(gy / gx)
    direction = np.zeros((w, h), int)
    for i in range(w):
        for j in range(h):
            theta = alpha[i, j]
            if abs(theta) <= 0.125 * math.pi or abs(theta) >= 0.875 * math.pi:
                direction[i, j] = 1
            elif 0.375 * math.pi <= abs(theta) <= 0.625 * math.pi:
                direction[i, j] = 3
            elif -0.375 * math.pi <= theta <= -0.125 * math.pi or 0.625 * math.pi <= abs(theta) <= 0.825 * math.pi:
                direction[i, j] = 4
            else:
                direction[i, j] = 2
    ud = np.zeros((w + 2, h + 2), float)
    ud[1:-1, 1:-1] = direction
    um = np.zeros((w + 2, h + 2), float)
    um[1:-1, 1:-1] = vm
    gn = np.copy(vm)
    for i in range(w):
        for j in range(h):
            unit = um[i:i+3, j:j+3]
            unit_dir = ud[i:i+3, j:j+3]
            index = np.where(unit_dir == direction[i, j])
            if len(np.where(vm[i, j] < unit[index])):
                gn[i, j] = 0
    gn_w = np.zeros((w, h), float)
    gn_h = np.zeros((w, h), float)
    idx_w = np.where(gn >= tw)
    idx_h = np.where(gn >= th)
    gn_w[idx_w] = gn[idx_w]
    gn_h[idx_h] = gn[idx_h]
    gn_w = gn_w - gn_h
    ugn = np.zeros((w + 2, h + 2), float)
    ugn[1:-1, 1:-1] = gn_w
    t_conn = np.zeros((w + 2, h + 2), float)
    for i in range(w):
        for j in range(h):
            if gn_h[i, j] > 0:
                t_conn[i:i+3, j:j+3] = ugn[i:i+3, j:j+3]
    gn_h = gn_h + t_conn[1:-1, 1:-1]
    if dst:
        cv2.imwrite(dst % 'Canny_result', gn_h)
    return gn_h


def zero_crossing(img, threshold, dst=None):
    w, h = img.shape
    result = np.zeros_like(img, np.uint8)
    uc = np.zeros((w + 2, h + 2), float)
    threshold = threshold * (np.max(img) - np.min(img)) / 255.0
    uc[1:-1, 1:-1] = img
    uc = uc - np.mean(img)
    for i in range(w):
        for j in range(h):
            unit = uc[i:i+3, j:j+3]
            if ((unit[1, 0] * unit[1, 2] < 0) and (abs(unit[1, 0] - unit[1, 2]) > threshold)) or \
                    ((unit[0, 1] * unit[2, 1] < 0) and (abs(unit[0, 1] - unit[2, 1]) > threshold)) or \
                    ((unit[0, 0] * unit[2, 2] < 0) and (abs(unit[0, 0] - unit[2, 2]) > threshold)) or \
                    ((unit[0, 2] * unit[2, 0] < 0) and (abs(unit[0, 2] - unit[2, 0]) > threshold)):
                result[i, j] = 255
    if dst:
        cv2.imwrite(dst % 'zero_crossing', result)
    return result


if __name__ == '__main__':
    pass
