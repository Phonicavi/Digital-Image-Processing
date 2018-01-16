import numpy as np
import math
import cmath
from utils import center_transform, normalize


def blurring_filter(img, a=0.1, b=0.1, t=1, dst=None):

    eps = 0.000001
    w, h = img.shape

    img_dft = np.fft.fft2(center_transform(img))
    hm = np.copy(img_dft)
    gm = np.copy(img_dft)

    for i in range(w):
        for j in range(h):
            theta = math.pi * (a * (i - w / 2) + b * (j - h / 2))
            hm[i, j] = 1 + 0.j if abs(theta) < eps else (t / theta) * math.sin(theta) * cmath.exp(-1.j * theta)
            gm[i, j] = img_dft[i, j] * hm[i, j]

    r_comp = np.real(np.fft.ifft2(gm))
    img_ct = center_transform(r_comp, dst)

    return img_ct, hm


def inverse_filter(img, hm, dst=None):

    img_dft = np.fft.fft2(center_transform(img))
    fm = img_dft / hm

    r_comp = np.real(np.fft.ifft2(fm))
    img_ct = center_transform(r_comp, dst)

    return img_ct


def wiener_deconv_filter(img, fm, hm, noise, dst=None):

    img_dft = np.fft.fft2(center_transform(img))
    sf = abs(np.fft.fft2(center_transform(fm))) ** 2
    sn = abs(np.fft.fft2(center_transform(noise))) ** 2
    result = ((abs(hm) ** 2) / hm / (abs(hm) ** 2 + sn / sf)) * img_dft

    r_comp = np.real(np.fft.ifft2(result))
    img_ct = center_transform(r_comp, dst)

    return img_ct


if __name__ == '__main__':
    pass
