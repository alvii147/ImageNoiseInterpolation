import numpy as np
from utils import (
    openImage,
    splitChannels,
    addNoise,
    combineChannels,
    saveImage,
)

if __name__ == '__main__':
    path = 'img/birb.png'
    dest = 'img/birb_noisy.png'

    # noise probability
    p = 0.5

    img = openImage(path)
    R, G, B = splitChannels(img)

    # add noise to all 3 channels
    R, is_noisy = addNoise(R, p)
    G, _ = addNoise(G, p, is_noisy)
    B, _ = addNoise(B, p, is_noisy)

    img = combineChannels(R, G, B)
    saveImage(img, dest)

    # save is_noisy arrays as binary files
    with open('data/is_noisy_birb.npy', 'wb') as npyfile:
        np.save(npyfile, is_noisy)