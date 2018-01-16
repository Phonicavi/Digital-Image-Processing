import cv2
import numpy as np


def description(imgs, dst=None):
    KK = len(imgs)
    w, h = imgs[0].shape
    vec = np.zeros((KK, w * h), np.uint8)
    for n in range(KK):
        vec[n, :] = imgs[n].flatten()

    mx = np.zeros((KK, 1), float)
    for k in range(w * h):
        mx = mx + np.reshape(vec[:, k], (KK, 1))
    mx /= w * h
    cx = np.zeros((KK, KK), float)
    for k in range(w * h):
        cx = cx + np.dot(vec[:, k], vec[:, k].T)
    cx /= w * h
    cx = cx - np.dot(mx, mx.T)

    D, V = np.linalg.eig(cx)
    V = V[::-1, :]
    pcs = 2
    vec_meta = np.zeros((w * h, pcs), float)
    for k in range(w * h):
        product = np.dot(np.reshape(V[0:pcs, :], (pcs, KK)), np.reshape(vec[:, k], (KK, 1)) - mx)
        vec_meta[k, :] = np.reshape(product, (1, pcs))
    vec_reco = np.zeros((w * h, KK), float)
    for k in range(w * h):
        product = np.dot(np.reshape(V[0:pcs, :], (pcs, KK)).T, np.reshape(vec_meta[k, :], (pcs, 1)))
        vec_reco[k, :] = np.reshape(product + mx, (1, KK))

    if dst:
        for n in range(KK):
            img = np.reshape(vec_reco[:, n], (w, h))
            cv2.imwrite(dst % ('restore-' + str(n+1)), img)

    return vec_reco


if __name__ == '__main__':
    pass
