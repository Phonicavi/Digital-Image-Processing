import cv2
import os
from utils import normalize, sharpen, mean_filter
from utils.laplacian import laplace_operation
from utils.sobel import sobel_operation


if __name__ == '__main__':

    fn = "./img/skeleton_orig.tif"
    if not os.path.isfile(fn):
        raise Exception('__invalid_input_image__')

    enh_laps = laplace_operation(fp=fn, dst="./img/laplace.tif")
    enh_sob = sobel_operation(fp=fn, dst="./img/sobel.tif")

    srp_laplace = sharpen(fp=fn, dst="./img/sharpen_laps.tif", enhancement=enh_laps)
    srp_sobel = sharpen(fp=fn, dst="./img/sharpen_sob.tif", enhancement=enh_sob)

    avg_filtered = mean_filter(enh_sob, size=5)
    cv2.imwrite("./img/avg_flt5.tif", normalize(avg_filtered))

    enh_mask = srp_laplace * avg_filtered
    cv2.imwrite("./img/enh_mask.tif", normalize(enh_mask))
    srp_mask = sharpen(fp=fn, dst="./img/sharpen_mask.tif", enhancement=enh_mask)

    srp_gamma = normalize(srp_mask ** 0.5)
    cv2.imwrite("./img/sharpen_gamma.tif", srp_gamma)
