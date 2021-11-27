import sys

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
USAGE += '\n\tinterpolate.py <SRC_PATH> <DEST_PATH>'

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

    # open image and split into RGB channels
    img = openImage(src)
    R, G, B = splitChannels(img)

    # get noisy pixels
    is_noisy_R = detectNoise(R, G, B, N)
    is_noisy_G = detectNoise(G, R, B, N)
    is_noisy_B = detectNoise(B, R, G, N)

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