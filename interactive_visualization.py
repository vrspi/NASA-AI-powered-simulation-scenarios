import plotly.express as px
from astropy.io import fits
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import logging

fits_file_path = "D:/AI Nasa/NEOSSAT_Data/NEOS_SCI_2024050000622.fits"

with fits.open(fits_file_path) as hdul:
    image_data = hdul[0].data

def create_interactive_plot(image_data, orbit):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Image", "Orbit Path"))

    # Original Image
    fig.add_trace(go.Image(z=image_data), row=1, col=1)

    if orbit is not None:
        # Orbit Path
        fig.add_trace(go.Scatter(x=orbit.x, y=orbit.y, mode='lines', name='Orbit'), row=1, col=2)
    else:
        # If orbit is None, add an empty scatter plot with a message
        fig.add_trace(go.Scatter(x=[], y=[], mode='markers', name='No Orbit Data'), row=1, col=2)
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text="Orbit data unavailable",
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=16, color="red")
        )

    fig.update_layout(title_text="Interactive Visualization")
    fig.show()

fig = px.imshow(image_data, color_continuous_scale='viridis')
fig.update_layout(title='Interactive NEOSSAT Image')
fig.show()