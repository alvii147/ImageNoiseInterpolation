import sys
import getopt

from utils import (
    openImage,
    splitChannels,
    slidingWindowOperation,
    combineChannels,
    saveImage,
)

# usage string
USAGE = 'Usage:'
USAGE += '\n\median.py src dest '
USAGE += '[-N window_length]'

if __name__ == '__main__':
    # exit if not enough arguments
    if len(sys.argv) < 3:
        print(USAGE)
        sys.exit(1)

    # image file source
    src = sys.argv[1]
    # image file destination
    dest = sys.argv[2]
    # window length
    N = 3

    # parse command line arguments
    try:
        opts, _ = getopt.getopt(sys.argv[3:], 'N:')
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-N':
            try:
                N = abs(int(arg))
            except:
                print('Error: N must be a positive integer')
                print(USAGE)
                sys.exit(1)

    # open image and split into RGB channels
    img = openImage(src)
    R, G, B = splitChannels(img)

    # apply median filter
    R = slidingWindowOperation(R, N, op='median')
    G = slidingWindowOperation(G, N, op='median')
    B = slidingWindowOperation(B, N, op='median')

    # combine RGB channels and save image
    img = combineChannels(R, G, B)
    saveImage(img, dest)