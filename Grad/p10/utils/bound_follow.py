import cv2
import numpy as np
from utils import mean_filter, binary_filter, start_point, rotate_from


def boundary(img, dst):
    w, h = img.shape
    ns_m = mean_filter(img, size_t=5, dst=dst)
    ns_b = binary_filter(ns_m, percent=0.9, dst=dst)

    x0, y0 = start_point(ns_b)
    b, c = np.array([x0, y0]), np.array([x0-1, y0])
    seq1 = np.zeros((2*(w+h), 2), np.uint)
    seq1[0, :], t = b, c
    while ns_b[c[0], c[1]] != 255:
        t = c
        c = rotate_from(b, t)
    seq1[1, :] = c
    b, c = c, t

    step = 2
    while (b[0] != x0) or (b[1] != y0):
        while ns_b[c[0], c[1]] != 255:
            t = c
            c = rotate_from(b, t)
        seq1[step, :] = c
        b, c = c, t
        step += 1

    bound = np.zeros((w, h), np.uint8)
    for k in range(2*(w+h)):
        p0 = seq1[k, :]
        bound[p0[0], p0[1]] = 255
        if ns_b[p0[0], p0[1]] != 255:
            break

    granularity = 25.0
    g_rows = np.floor(w / granularity)
    g_cols = np.floor(h / granularity)
    Gmax = np.zeros((w, h), np.uint8)
    Gmin = np.zeros((int(g_rows), int(g_cols)), np.uint8)
    for i in range(w):
        for j in range(h):
            if bound[i, j] == 255:
                grid_x = np.floor(g_rows * i / float(w))
                grid_y = np.floor(g_cols * j / float(h))
                Gmax_x = np.floor(grid_x / g_rows * float(w))
                Gmax_y = np.floor(grid_y / g_cols * float(h))
                Gmin[int(grid_x), int(grid_y)] = 255
                Gmax[int(Gmax_x), int(Gmax_y)] = 255

    if dst:
        cv2.imwrite(dst % 'noisy_outline', bound)
        cv2.imwrite(dst % 'noisy_Gmin', Gmin)
        cv2.imwrite(dst % 'noisy_Gmax', Gmax)
    return Gmax


if __name__ == '__main__':
    pass

