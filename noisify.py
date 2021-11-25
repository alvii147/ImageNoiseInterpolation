import sys
import getopt

from utils import (
    openImage,
    splitChannels,
    addNoise,
    combineChannels,
    saveImage,
)

# usage string
USAGE = 'Usage:'
USAGE += '\n\tnoisify.py src dest '
USAGE += '[-p noise_density]'

if __name__ == '__main__':
    # exit if not enough arguments
    if len(sys.argv) < 3:
        print(USAGE)
        sys.exit(1)

    # image file source
    src = sys.argv[1]
    # image file destination
    dest = sys.argv[2]
    # noise density
    p = 0.5

    # parse command line arguments
    try:
        opts, _ = getopt.getopt(sys.argv[3:], 'p:')
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-p':
            try:
                p = float(arg)
                if p < 0 or p > 1:
                    raise ValueError
            except:
                print('Error: p must be a float in range [0, 1]')
                print(USAGE)
                sys.exit(1)

    # open image and split into RGB channels
    img = openImage(src)
    R, G, B = splitChannels(img)

    # add noise to all 3 channels
    R, is_noisy = addNoise(R, p)
    G, _ = addNoise(G, p, is_noisy)
    B, _ = addNoise(B, p, is_noisy)

    # combine RGB channels and save image
    img = combineChannels(R, G, B)
    saveImage(img, dest)