import sys
import getopt

from utils import (
    openImage,
    splitChannels,
    detectNoise,
    interpolateChannel,
    combineChannels,
    saveImage,
)

# usage string
USAGE = 'Usage:'
USAGE += '\n\tinterpolate.py src dest '
USAGE += '[-E noise_threshold]'

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
    # noise threshold
    E = 53

    # parse command line arguments
    try:
        opts, _ = getopt.getopt(sys.argv[3:], 'E:')
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-E':
            try:
                E = abs(float(arg))
            except:
                print('Error: E must be a float in range [0, 255]')
                print(USAGE)
                sys.exit(1)

    # open image and split into RGB channels
    img = openImage(src)
    R, G, B = splitChannels(img)

    # get noisy pixels
    is_noisy_R = detectNoise(R, N, E=E)
    is_noisy_G = detectNoise(G, N, E=E)
    is_noisy_B = detectNoise(B, N, E=E)

    # interpolate noisy pixels
    R_interpolated = interpolateChannel(
        R,
        G,
        B,
        is_noisy_R,
        is_noisy_G,
        is_noisy_B,
    )

    G_interpolated = interpolateChannel(
        G,
        R,
        B,
        is_noisy_G,
        is_noisy_R,
        is_noisy_B,
    )

    B_interpolated = interpolateChannel(
        B,
        R,
        G,
        is_noisy_B,
        is_noisy_R,
        is_noisy_G,
    )

    # combine RGB channels and save image
    img = combineChannels(
        R_interpolated,
        G_interpolated,
        B_interpolated,
    )
    saveImage(img, dest)