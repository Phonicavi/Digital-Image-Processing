import cv2
import pywt


def dwt(img, mode='haar', dst=None):
    cc = pywt.wavedec2(data=img, wavelet=mode, level=3)
    rec = pywt.waverec2(coeffs=cc, wavelet=mode)
    diff = img - rec
    if dst:
        cv2.imwrite(dst % (mode + '_rec'), rec)
        cv2.imwrite(dst % (mode + '_diff'), diff)
    return rec, diff


if __name__ == '__main__':
    pass
