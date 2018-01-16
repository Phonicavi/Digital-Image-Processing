import os
import utils.hist as hist
import utils.benchmark as bcm

if __name__ == '__main__':

    dir = "./img/"
    fp = ["Fig1", "Fig2"]
    levels = 100

    for fn in fp:

        src = os.path.join(dir, "%s.jpg" % fn)
        dest = os.path.join(dir, "%s_hist_eq.jpg" % fn)
        dest_bcm = os.path.join(dir, "%s_hist_eq-Benchmark.jpg" % fn)

        # origin
        hist.histogram(src, levels)

        # evaluation: custom & standard
        hist.equalize(src, dest)
        bcm.equalize(src, dest_bcm)

        hist.histogram(dest, levels)
        hist.histogram(dest_bcm, levels)
