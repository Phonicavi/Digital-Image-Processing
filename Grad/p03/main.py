from utils.ideal_filter import ideal_filter
from utils.butterworth_filter import *
from utils.gaussian_filter import *


if __name__ == '__main__':

    fn = "./img/characters_test_pattern.tif"
    opt = parse_args()

    opt.mode = 'hi'
    ideal_filter(fp=fn, param=opt, dst="./img/ideal_filter_hi.tif")
    butterworth_filter(fp=fn, param=opt, dst="./img/butterworth_filter_hi.tif")
    gaussian_filter(fp=fn, param=opt, dst="./img/gaussian_filter_hi.tif")

    opt.mode = 'lo'
    ideal_filter(fp=fn, param=opt, dst="./img/ideal_filter_lo.tif")
    butterworth_filter(fp=fn, param=opt, dst="./img/butterworth_filter_lo.tif")
    gaussian_filter(fp=fn, param=opt, dst="./img/gaussian_filter_lo.tif")
