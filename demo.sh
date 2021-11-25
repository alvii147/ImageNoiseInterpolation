#!/usr/bin/bash

ORIGINAL_IMG=img/birb.png
NOISY_IMG=img/birb_noisy.png
INTERPOLATED_IMG=img/birb_interpolated.png
MEDIAN_IMG=img/birb_median.png

python noisify.py $ORIGINAL_IMG $NOISY_IMG -p 0.5
python interpolate.py $NOISY_IMG $INTERPOLATED_IMG -E 15
python median.py $NOISY_IMG $MEDIAN_IMG -N 3