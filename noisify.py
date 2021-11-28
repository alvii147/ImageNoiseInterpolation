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
USAGE += '\n\tnoisify.py <SRC_PATH> <DEST_PATH>'
USAGE += '[-p POSITIVE_IMPULSE_DENSITY] [-n NEGATIVE_IMPULSE_DENSITY]'

if __name__ == '__main__':
    # exit if not enough arguments
    if len(sys.argv) < 3:
        print(USAGE)
        sys.exit(1)

    # image file source
    src = sys.argv[1]
    # image file destination
    dest = sys.argv[2]
    # impulsive noise densities
    p = 0.02
    n = 0.02

    # parse command line arguments
    try:
        opts, _ = getopt.getopt(sys.argv[3:], 'p:n:')
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-p':
            try:
                p = float(arg)
            except:
                print('Error: p must be a float in range [0, 1)')
                print(USAGE)
                sys.exit(1)
        elif opt == '-n':
            try:
                n = float(arg)
            except:
                print('Error: n must be a float in range [0, 1)')
                print(USAGE)
                sys.exit(1)

    # open image and split into RGB channels
    img = openImage(src)
    R, G, B = splitChannels(img)

    # add noise to all 3 channels
    R, _ = addNoise(R, p1=p, p2=n)
    G, _ = addNoise(G, p1=p, p2=n)
    B, _ = addNoise(B, p1=p, p2=n)

    # combine RGB channels and save image
    img = combineChannels(R, G, B)
    saveImage(img, dest)