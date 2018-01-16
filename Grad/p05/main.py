import cv2
import os

from utils import get_gaussian_noise, truncate
from utils.filter import blurring_filter, inverse_filter, wiener_deconv_filter


if __name__ == '__main__':

    fn = "./img/book_cover.jpg"
    if not os.path.exists(fn):
        raise Exception('[Error] image file not exists')

    img = cv2.cvtColor(cv2.imread(fn, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    img_blurred, hm = blurring_filter(img, a=0.1, b=0.1, t=1, dst="./img/blurred.tif")

    noise_gaussian = get_gaussian_noise(img_blurred, a=0, b=650**0.5)
    img_noise_gauss = truncate(img_blurred + noise_gaussian, dst="./img/blurred_noise_g.tif")

    img_inverse_blurred = inverse_filter(img_blurred, hm, dst="./img/inverse_blurred.tif")
    img_inverse_blurred_g = inverse_filter(img_noise_gauss, hm, dst="./img/inverse_noise_g.tif")
    img_weiner_blurred = wiener_deconv_filter(img_blurred, img, hm, noise_gaussian, dst="./img/wiener_blurred.tif")
    img_weiner_blurred_g = wiener_deconv_filter(img_noise_gauss, img, hm, noise_gaussian, dst="./img/wiener_noise_g.tif")

    noise_gaussian = get_gaussian_noise(img_blurred, a=0, b=65**0.5)
    img_noise_gauss = truncate(img_blurred + noise_gaussian, dst="./img/blurred_noise_g2.tif")

    img_inverse_blurred = inverse_filter(img_blurred, hm, dst="./img/inverse_blurred2.tif")
    img_inverse_blurred_g = inverse_filter(img_noise_gauss, hm, dst="./img/inverse_noise_g.tif")
    img_weiner_blurred = wiener_deconv_filter(img_blurred, img, hm, noise_gaussian, dst="./img/wiener_blurred2.tif")
    img_weiner_blurred_g = wiener_deconv_filter(img_noise_gauss, img, hm, noise_gaussian, dst="./img/wiener_noise_g2.tif")

    noise_gaussian = get_gaussian_noise(img_blurred, a=0, b=0.65**0.5)
    img_noise_gauss = truncate(img_blurred + noise_gaussian, dst="./img/blurred_noise_g3.tif")

    img_inverse_blurred = inverse_filter(img_blurred, hm, dst="./img/inverse_blurred3.tif")
    img_inverse_blurred_g = inverse_filter(img_noise_gauss, hm, dst="./img/inverse_noise_g.tif")
    img_weiner_blurred = wiener_deconv_filter(img_blurred, img, hm, noise_gaussian, dst="./img/wiener_blurred3.tif")
    img_weiner_blurred_g = wiener_deconv_filter(img_noise_gauss, img, hm, noise_gaussian, dst="./img/wiener_noise_g3.tif")
