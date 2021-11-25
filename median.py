from utils import (
    openImage,
    splitChannels,
    slidingWindowOperation,
    combineChannels,
    saveImage,
)

if __name__ == '__main__':
    path = 'img/birb_noisy.png'
    dest = 'img/birb_median.png'

    img = openImage(path)
    R, G, B = splitChannels(img)

    # apply median filter
    N = 3
    R = slidingWindowOperation(R, N, op='median')
    G = slidingWindowOperation(G, N, op='median')
    B = slidingWindowOperation(B, N, op='median')

    img = combineChannels(R, G, B)
    saveImage(img, dest)