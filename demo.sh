#!/usr/bin/bash

ORIGINAL_IMG=img/birb.png
NOISY_IMG=img/birb_noisy.png
INTERPOLATED_IMG=img/birb_interpolated.png
MEDIAN_IMG=img/birb_median.png

echo Adding noise to $ORIGINAL_IMG
python noisify.py $ORIGINAL_IMG $NOISY_IMG -p 0.47 -n 0.47
echo Applying interpolation on $NOISY_IMG
python interpolate.py $NOISY_IMG $INTERPOLATED_IMG -E 15
echo Applying median filter on $NOISY_IMG
python median.py $NOISY_IMG $MEDIAN_IMG -N 3