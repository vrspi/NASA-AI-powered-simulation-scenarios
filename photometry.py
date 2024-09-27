from astropy.io import fits
import numpy as np

def measure_brightness(image_data):
    return np.sum(image_data)

def generate_light_curve(fits_files):
    light_curve = []
    for file in fits_files:
        with fits.open(file) as hdul:
            data = hdul[0].data
            brightness = measure_brightness(data)
            light_curve.append(brightness)
    return light_curve