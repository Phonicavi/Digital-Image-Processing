import cv2
import os

from utils import parse_args, get_noise, truncate
from utils.filter import arithmetic_mean_filter, geometric_mean_filter
from utils.filter import contraharmonic_mean_filter
from utils.filter import min_filter, max_filter
from utils.filter import median_filter, alpha_mean_filter


if __name__ == '__main__':

    fp = "./img/Circuit.tif"
    if not os.path.exists(fp):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    w, h = img.shape
    opt = parse_args()
    opt.width = w
    opt.height = h

    opt.a = 0
    opt.b = 20
    opt.mode = "gaussian"
    noise_gaussian = get_noise(param=opt)
    img_noise_gaussian = truncate(img + noise_gaussian, dst="./img/gaussian_noise.tif")
    img_gaussian_arifi = arithmetic_mean_filter(img_noise_gaussian, size=(3, 3))
    cv2.imwrite("./img/gaussian_arithmetic.tif", img_gaussian_arifi)
    img_gaussian_geofi = geometric_mean_filter(img_noise_gaussian, size=(3, 3))
    cv2.imwrite("./img/gaussian_geometric.tif", img_gaussian_geofi)

    opt.a = 0
    opt.b = 800 ** 0.5
    opt.mode = "uniform"
    noise_uniform = get_noise(param=opt)
    img_noise_uniform = truncate(img + noise_uniform, dst="./img/uniform_noise.tif")

    opt.a = 0.1
    opt.b = 0.1
    opt.mode = "salt"
    noise_salt = get_noise(param=opt)
    img_noise_salt = truncate(img + noise_salt, dst="./img/salt_noise.tif")
    img_salt_contrharm = contraharmonic_mean_filter(img_noise_salt, size=(3, 3), param=1.5)
    cv2.imwrite("./img/salt_contraharmonic.tif", img_salt_contrharm)
    img_salt_min = min_filter(img_noise_salt, size=(3, 3))
    cv2.imwrite("./img/salt_min.tif", img_salt_min)

    opt.mode = "pepper"
    noise_pepper = get_noise(param=opt)
    img_noise_pepper = truncate(img + noise_pepper, dst="./img/pepper_noise.tif")
    img_pepper_contrharm = contraharmonic_mean_filter(img_noise_pepper, size=(3, 3), param=1.5)
    cv2.imwrite("./img/pepper_contraharmonic.tif", img_pepper_contrharm)
    img_pepper_max = max_filter(img_noise_pepper, size=(3, 3))
    cv2.imwrite("./img/pepper_max.tif", img_pepper_max)

    opt.mode = "salt & pepper"
    noise_salt8pepper = get_noise(param=opt)
    img_noise_unisp = truncate(img_noise_uniform + noise_salt8pepper, dst="./img/salt&pepper_noise.tif")
    img_unisp_arifi = arithmetic_mean_filter(img_noise_unisp, size=(3, 3))
    cv2.imwrite("./img/unisp_arithmetic.tif", img_unisp_arifi)
    img_unisp_geofi = geometric_mean_filter(img_noise_unisp, size=(3, 3))
    cv2.imwrite("./img/unisp_geometric.tif", img_unisp_geofi)
    img_salt8pepper_median = median_filter(img_noise_unisp, size=(3, 3))
    cv2.imwrite("./img/salt&pepper_median.tif", img_salt8pepper_median)
    img_salt8pepper_med2 = median_filter(img_salt8pepper_median, size=(3, 3))
    cv2.imwrite("./img/salt&pepper_median2.tif", img_salt8pepper_med2)
    img_salt8pepper_med3 = median_filter(img_salt8pepper_med2, size=(3, 3))
    cv2.imwrite("./img/salt&pepper_median3.tif", img_salt8pepper_med3)

    img_salt8pepper_alpha = alpha_mean_filter(img_noise_unisp, size=(3, 3), param=5)
    cv2.imwrite("./img/salt&pepper_alpha.tif", img_salt8pepper_alpha)

