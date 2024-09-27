import plotly.express as px
from astropy.io import fits
import numpy as np

fits_file_path = "D:/AI Nasa/NEOSSAT_Data/NEOS_SCI_2024050000622.fits"

with fits.open(fits_file_path) as hdul:
    image_data = hdul[0].data

fig = px.imshow(image_data, color_continuous_scale='viridis')
fig.update_layout(title='Interactive NEOSSAT Image')
fig.show()