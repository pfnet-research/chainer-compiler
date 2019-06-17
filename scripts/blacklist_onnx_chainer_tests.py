import os
import shutil


BLACKLISTS = {
    10: ['deconvolution2d_group3', 'deconvolutionnd_group3', 'resizeimages']
}


def main():
    os.chdir(os.path.dirname(__file__))
    for opset_version, names in BLACKLISTS.items():
        for name in names:
            shutil.rmtree('../out/opset%d/test_%s' % (opset_version, name))


main()
