import plotly.graph_objects as go
from astropy.io import fits
import numpy as np

fits_file_path = "D:/AI Nasa/NEOSSAT_Data/NEOS_SCI_2024050000622.fits"

with fits.open(fits_file_path) as hdul:
    image_data = hdul[0].data

x, y = np.meshgrid(range(image_data.shape[1]), range(image_data.shape[0]))
z = image_data

fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='3D NEOSSAT Image', autosize=False,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()