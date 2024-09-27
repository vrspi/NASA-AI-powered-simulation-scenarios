from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import cv2
import logging

# Set up logging
log_file = 'visualization_debug.txt'
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=log_file,
                    filemode='w')

# Add a console handler to display logs in the console as well
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def load_fits_data(fits_file_path):
    logging.info(f"Attempting to load FITS file: {fits_file_path}")
    if not os.path.exists(fits_file_path):
        logging.error(f"File not found at {fits_file_path}")
        raise FileNotFoundError(f"Error: File not found at {fits_file_path}")
    
    try:
        with fits.open(fits_file_path) as hdul:
            image_data = hdul[0].data
            header = hdul[0].header
        logging.info(f"Successfully loaded FITS file: {fits_file_path}")
        return image_data, header
    except Exception as e:
        logging.error(f"Error loading FITS file: {e}")
        raise

def apply_gaussian_filter(image_data, kernel_size=5, sigma=1):
    logging.debug(f"Applying Gaussian filter with kernel size {kernel_size} and sigma {sigma}")
    return cv2.GaussianBlur(image_data, (kernel_size, kernel_size), sigma)

def apply_median_filter(image_data, kernel_size=5):
    logging.debug(f"Applying Median filter with kernel size {kernel_size}")
    return cv2.medianBlur(image_data, kernel_size)

def apply_histogram_equalization(image_data):
    logging.debug("Applying histogram equalization")
    return cv2.equalizeHist(image_data.astype(np.uint8))

def apply_clahe(image_data, clip_limit=2.0, tile_grid_size=(8,8)):
    logging.debug(f"Applying CLAHE with clip limit {clip_limit} and tile grid size {tile_grid_size}")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_data.astype(np.uint8))

def detect_edges(image_data):
    logging.debug("Detecting edges")
    edges = cv2.Canny(image_data, threshold1=100, threshold2=200)
    return edges

def apply_threshold(image_data, threshold=128):
    logging.debug(f"Applying threshold with value {threshold}")
    _, binary = cv2.threshold(image_data, threshold, 255, cv2.THRESH_BINARY)
    return binary

def apply_adaptive_threshold(image_data):
    logging.debug("Applying adaptive threshold")
    binary = cv2.adaptiveThreshold(image_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return binary

def highlight_potential_neos(image_data, threshold_percentile=99):
    # Convert to 8-bit image
    image_8bit = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # Apply threshold
    threshold_value = np.percentile(image_8bit, threshold_percentile)
    _, binary = cv2.threshold(image_8bit, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around contours
    result = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return result

def visualize_fits(fits_file_path, cmap='viridis', norm=LogNorm()):
    logging.info(f"Visualizing FITS file: {fits_file_path}")
    image_data, header = load_fits_data(fits_file_path)
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    im1 = ax1.imshow(image_data, cmap=cmap, norm=norm)
    plt.colorbar(im1, ax=ax1, label='Flux')
    ax1.set_title(f'Original NEOSSAT Image: {os.path.basename(fits_file_path)}')
    ax1.set_xlabel('Pixel X')
    ax1.set_ylabel('Pixel Y')
    
    # Add some metadata from the header
    ax1.text(0.02, 0.98, f"Date: {header.get('DATE-OBS', 'N/A')}", transform=ax1.transAxes, va='top')
    ax1.text(0.02, 0.95, f"Exposure Time: {header.get('EXPTIME', 'N/A')} s", transform=ax1.transAxes, va='top')
    
    # Highlighted potential NEOs
    highlighted_image = highlight_potential_neos(image_data)
    ax2.imshow(highlighted_image)
    ax2.set_title('Potential NEO Candidates')
    ax2.set_xlabel('Pixel X')
    ax2.set_ylabel('Pixel Y')
    
    plt.tight_layout()
    logging.info("Displaying the visualized FITS image with potential NEO highlights")
    plt.show()

def plot_histogram(fits_file_path, bins=100):
    logging.info(f"Plotting histogram for FITS file: {fits_file_path}")
    image_data, _ = load_fits_data(fits_file_path)
    
    plt.figure(figsize=(10, 6))
    plt.hist(image_data.flatten(), bins=bins, log=True)
    plt.title(f'Histogram of Pixel Values: {os.path.basename(fits_file_path)}')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    logging.info("Displaying the histogram")
    plt.show()

if __name__ == "__main__":
    fits_file_path = "D:/AI Nasa/NEOSSAT_Data/NEOS_SCI_2024050000622.fits"
    
    try:
        logging.info("Starting visualization process")
        visualize_fits(fits_file_path)
        plot_histogram(fits_file_path)
        logging.info("Visualization process completed successfully")
    except Exception as e:
        logging.exception(f"An error occurred during visualization: {e}")
        print(f"An error occurred: {e}")
        print("Please ensure the file is a valid FITS file and not corrupted.")
        print(f"Check the log file '{log_file}' for detailed debug information.")