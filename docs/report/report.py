import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

# /////////////////////////////////////////////////////////////////////////

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    openImage,
    splitChannels,
    addNoise,
    detectNoise,
    interpolateChannel,
    combineChannels,
)

# /////////////////////////////////////////////////////////////////////////

img = openImage('birb.png')
R, G, B = splitChannels(img)

# /////////////////////////////////////////////////////////////////////////

R, G, B = splitChannels(img)
Z = np.zeros(R.shape, dtype=np.uint8)

R_img = combineChannels(R, Z, Z)
B_img = combineChannels(Z, G, Z)
G_img = combineChannels(Z, Z, B)

# /////////////////////////////////////////////////////////////////////////

fig, axis = plt.subplots(2, 2)
fig.set_figheight(7)
fig.set_figwidth(7)

axis[0, 0].imshow(img, interpolation='nearest')
axis[0, 0].set_title('Original Image')

axis[0, 1].imshow(R_img, interpolation='nearest')
axis[0, 1].set_title('Channel R')

axis[1, 0].imshow(B_img, interpolation='nearest')
axis[1, 0].set_title('Channel R')

axis[1, 1].imshow(G_img, interpolation='nearest')
axis[1, 1].set_title('Channel B')

plt.savefig('img/clean_plot.png')

# /////////////////////////////////////////////////////////////////////////

p1 = 0.02
p2 = 0.02

R_noisy, is_noisy_R = addNoise(R, p1=p1, p2=p2)
G_noisy, is_noisy_G = addNoise(G, p1=p1, p2=p2)
B_noisy, is_noisy_B = addNoise(B, p1=p1, p2=p2)

noisy_img = combineChannels(R_noisy, G_noisy, B_noisy)
R_noisy_img = combineChannels(R_noisy, Z, Z)
G_noisy_img = combineChannels(Z, G_noisy, Z)
B_noisy_img = combineChannels(Z, Z, B_noisy)

# /////////////////////////////////////////////////////////////////////////

fig, axis = plt.subplots(2, 2)
fig.set_figheight(7)
fig.set_figwidth(7)

axis[0, 0].imshow(noisy_img, interpolation='nearest')
axis[0, 0].set_title('Impulsive Noise')

axis[0, 1].imshow(R_noisy_img, interpolation='nearest')
axis[0, 1].set_title('Impulsive Noise (R)')

axis[1, 0].imshow(G_noisy_img, interpolation='nearest')
axis[1, 0].set_title('Impulsive Noise (G)')

axis[1, 1].imshow(B_noisy_img, interpolation='nearest')
axis[1, 1].set_title('Impulsive Noise (B)')

plt.savefig('img/noisy_plot.png')

# /////////////////////////////////////////////////////////////////////////

detected_noise_R = detectNoise(R_noisy, G_noisy, B_noisy)
detected_noise_G = detectNoise(G_noisy, R_noisy, B_noisy)
detected_noise_B = detectNoise(B_noisy, R_noisy, G_noisy)

# /////////////////////////////////////////////////////////////////////////

fig, axis = plt.subplots(2, 3)
fig.set_figheight(6)
fig.set_figwidth(12)

axis[0, 0].imshow(is_noisy_R, cmap='magma', interpolation='nearest')
axis[0, 0].set_title('Actual Noise (R)')

axis[0, 1].imshow(is_noisy_G, cmap='magma', interpolation='nearest')
axis[0, 1].set_title('Actual Noise (G)')

axis[0, 2].imshow(is_noisy_B, cmap='magma', interpolation='nearest')
axis[0, 2].set_title('Actual Noise (B)')

axis[1, 0].imshow(detected_noise_R, cmap='cividis', interpolation='nearest')
axis[1, 0].set_title('Detected Noise (R)')

axis[1, 1].imshow(detected_noise_G, cmap='cividis', interpolation='nearest')
axis[1, 1].set_title('Detected Noise (G)')

axis[1, 2].imshow(detected_noise_B, cmap='cividis', interpolation='nearest')
axis[1, 2].set_title('Detected Noise (B)')

plt.savefig('img/noise_plot.png')

# /////////////////////////////////////////////////////////////////////////

R_interpolated = interpolateChannel(
    R_noisy,
    G_noisy,
    B_noisy,
    detected_noise_R,
    detected_noise_G,
    detected_noise_B,
)

G_interpolated = interpolateChannel(
    G_noisy,
    R_noisy,
    B_noisy,
    detected_noise_G,
    detected_noise_R,
    detected_noise_B,
)

B_interpolated = interpolateChannel(
    B_noisy,
    R_noisy,
    G_noisy,
    detected_noise_B,
    detected_noise_R,
    detected_noise_G,
)

interpolated_img = combineChannels(
    R_interpolated,
    G_interpolated,
    B_interpolated,
)

Z = np.zeros(R_interpolated.shape, dtype=np.uint8)
R_interpolated_img = combineChannels(R_interpolated, Z, Z)
G_interpolated_img = combineChannels(Z, G_interpolated, Z)
B_interpolated_img = combineChannels(Z, Z, B_interpolated)

# /////////////////////////////////////////////////////////////////////////

fig, axis = plt.subplots(1, 4)
fig.set_figheight(3)
fig.set_figwidth(12)

axis[0].imshow(interpolated_img, interpolation='nearest')
axis[0].set_title('Interpolated')

axis[1].imshow(R_interpolated_img, interpolation='nearest')
axis[1].set_title('Interpolated (R)')

axis[2].imshow(G_interpolated_img, interpolation='nearest')
axis[2].set_title('Interpolated (G)')

axis[3].imshow(B_interpolated_img, interpolation='nearest')
axis[3].set_title('Interpolated (B)')

plt.savefig('img/interpolated_plot.png')

# /////////////////////////////////////////////////////////////////////////

fig, axis = plt.subplots(2, 4)
fig.set_figheight(6)
fig.set_figwidth(12)

axis[0, 0].imshow(noisy_img, interpolation='nearest')
axis[0, 0].set_title('Impulsive Noise')

axis[0, 1].imshow(R_noisy_img, interpolation='nearest')
axis[0, 1].set_title('Impulsive Noise (R)')

axis[0, 2].imshow(G_noisy_img, interpolation='nearest')
axis[0, 2].set_title('Impulsive Noise (G)')

axis[0, 3].imshow(B_noisy_img, interpolation='nearest')
axis[0, 3].set_title('Impulsive Noise (B)')

axis[1, 0].imshow(interpolated_img, interpolation='nearest')
axis[1, 0].set_title('Interpolated')

axis[1, 1].imshow(R_interpolated_img, interpolation='nearest')
axis[1, 1].set_title('Interpolated (R)')

axis[1, 2].imshow(G_interpolated_img, interpolation='nearest')
axis[1, 2].set_title('Interpolated (G)')

axis[1, 3].imshow(B_interpolated_img, interpolation='nearest')
axis[1, 3].set_title('Interpolated (B)')

plt.savefig('img/comparison_plot.png')