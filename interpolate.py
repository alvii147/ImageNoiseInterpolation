from utils import (
    openImage,
    splitChannels,
    detectNoise,
    interpolateChannel,
    combineChannels,
    saveImage,
)

if __name__ == '__main__':
    path = 'img/birb_noisy.png'
    dest = 'img/birb_interpolated.png'

    img = openImage(path)
    R, G, B = splitChannels(img)

    # get noisy pixels
    N = 3
    E = 12
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

    img = combineChannels(
        R_interpolated,
        G_interpolated,
        B_interpolated,
    )

    saveImage(img, dest)