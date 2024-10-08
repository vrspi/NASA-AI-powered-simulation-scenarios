from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import os
import cv2
import logging
import json
from photometry import measure_brightness, generate_light_curve
from database_integration import create_database, insert_data
from astrometry import calculate_orbit, Orbit
from interactive_visualization import create_interactive_plot
from astropy.coordinates import SkyCoord
from astropy.time import Time
import shutil

# Set up logging
log_file = 'visualization_debug.txt'
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file, mode='w'),
                        logging.StreamHandler()
                    ])

# Progress file path
progress_file = 'progress.json'

def load_fits_data(fits_file_path):
    logging.info(f"Attempting to load FITS file: {fits_file_path}")
    if not os.path.exists(fits_file_path):
        logging.error(f"File not found at {fits_file_path}")
        raise FileNotFoundError(f"Error: File not found at {fits_file_path}")
    
    try:
        with fits.open(fits_file_path) as hdul:
            logging.debug("Opened FITS file successfully.")
            image_data = hdul[0].data
            header = hdul[0].header
            logging.debug("Extracted image data and header from FITS file.")
        logging.info(f"Successfully loaded FITS file: {fits_file_path}")
        return image_data, header
    except Exception as e:
        logging.error(f"Error loading FITS file: {e}")
        raise

def apply_gaussian_filter(image_data, kernel_size=5, sigma=1):
    logging.debug(f"Applying Gaussian filter with kernel size {kernel_size} and sigma {sigma}")
    try:
        filtered_image = cv2.GaussianBlur(image_data, (kernel_size, kernel_size), sigma)
        logging.debug("Gaussian filter applied successfully.")
        return filtered_image
    except Exception as e:
        logging.error(f"Error applying Gaussian filter: {e}")
        raise

def apply_median_filter(image_data, kernel_size=5):
    logging.debug(f"Applying Median filter with kernel size {kernel_size}")
    try:
        filtered_image = cv2.medianBlur(image_data, kernel_size)
        logging.debug("Median filter applied successfully.")
        return filtered_image
    except Exception as e:
        logging.error(f"Error applying Median filter: {e}")
        raise

def apply_histogram_equalization(image_data):
    logging.debug("Applying histogram equalization")
    try:
        equalized_image = cv2.equalizeHist(image_data.astype(np.uint8))
        logging.debug("Histogram equalization applied successfully.")
        return equalized_image
    except Exception as e:
        logging.error(f"Error applying histogram equalization: {e}")
        raise

def apply_clahe(image_data, clip_limit=2.0, tile_grid_size=(8,8)):
    logging.debug(f"Applying CLAHE with clip limit {clip_limit} and tile grid size {tile_grid_size}")
    try:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        clahe_image = clahe.apply(image_data.astype(np.uint8))
        logging.debug("CLAHE applied successfully.")
        return clahe_image
    except Exception as e:
        logging.error(f"Error applying CLAHE: {e}")
        raise

def detect_edges(image_data):
    logging.debug("Detecting edges")
    try:
        edges = cv2.Canny(image_data, threshold1=100, threshold2=200)
        logging.debug("Edge detection completed successfully.")
        return edges
    except Exception as e:
        logging.error(f"Error detecting edges: {e}")
        raise

def apply_threshold(image_data, threshold=128):
    logging.debug(f"Applying threshold with value {threshold}")
    try:
        _, binary = cv2.threshold(image_data, threshold, 255, cv2.THRESH_BINARY)
        logging.debug("Threshold applied successfully.")
        return binary
    except Exception as e:
        logging.error(f"Error applying threshold: {e}")
        raise

def apply_adaptive_threshold(image_data):
    logging.debug("Applying adaptive threshold")
    try:
        binary = cv2.adaptiveThreshold(image_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        logging.debug("Adaptive threshold applied successfully.")
        return binary
    except Exception as e:
        logging.error(f"Error applying adaptive threshold: {e}")
        raise

def highlight_potential_neos(image_data, threshold_percentile=99):
    logging.debug(f"Highlighting potential NEOs with threshold percentile {threshold_percentile}")
    try:
        # Convert to 8-bit image
        image_8bit = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        logging.debug("Image normalized to 8-bit successfully.")
        
        # Apply threshold
        threshold_value = np.percentile(image_8bit, threshold_percentile)
        logging.debug(f"Calculated threshold value: {threshold_value}")
        _, binary = cv2.threshold(image_8bit, threshold_value, 255, cv2.THRESH_BINARY)
        logging.debug("Threshold applied for NEO highlighting.")
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logging.debug(f"Found {len(contours)} contours.")
        
        # Draw rectangles around contours
        result = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        logging.debug("Rectangles drawn around potential NEOs.")
        
        return result
    except Exception as e:
        logging.error(f"Error highlighting potential NEOs: {e}")
        raise

def update_progress(current, total, filename, status, brightness, orbit=None):
    progress = {
        "total_files": total,
        "current_file": current,
        "filename": filename,
        "status": status,
        "brightness": brightness,
        "orbit": orbit.__dict__ if isinstance(orbit, Orbit) else orbit
    }
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=4)
        logging.debug(f"Progress updated: {progress}")
    except Exception as e:
        logging.error(f"Error updating progress file: {e}")

def visualize_fits(fits_file_path, cmap='viridis', norm=LogNorm(), save_fig=False, fig_path=None,
                  threshold_percentile=99, kernel_size=5, sigma=1):
    logging.info(f"Visualizing FITS file: {fits_file_path}")
    try:
        image_data, header = load_fits_data(fits_file_path)
        
        # Measure brightness using photometry module
        brightness = measure_brightness(image_data)
        logging.info(f"Measured brightness: {brightness}")
        
        # Insert data into the database
        filename = os.path.basename(fits_file_path)
        date_obs = header.get('DATE-OBS', 'N/A')
        exptime = header.get('EXPTIME', 'N/A')
        logging.debug(f"Inserting data into database: filename={filename}, date_obs={date_obs}, exptime={exptime}, brightness={brightness}")
        insert_data('neossat_data.db', (filename, date_obs, exptime, brightness, ''))
        logging.debug("Data inserted into database successfully.")
        
        # Calculate orbit using astrometry module
        ra = header.get('RA')
        dec = header.get('DEC')
        ra_sub = header.get('RA_SUB', ra)  # Placeholder if RA_SUB not present
        dec_sub = header.get('DEC_SUB', dec)  # Placeholder if DEC_SUB not present
        date_obs_str = header.get('DATE-OBS', None)
        
        if ra is None or dec is None or date_obs_str is None:
            logging.warning("Missing RA, DEC, or DATE-OBS in FITS header. Orbit calculation may be inaccurate.")
            initial_coords = SkyCoord(0, 0, unit='deg')  # Default to (0,0) if missing
            subsequent_coords = SkyCoord(0, 0, unit='deg')
            times = Time('J2000')  # Default time
        else:
            initial_coords = SkyCoord(ra, dec, unit='deg')
            subsequent_coords = SkyCoord(ra_sub, dec_sub, unit='deg')
            times = Time(date_obs_str)
        
        logging.debug(f"Initial Coordinates: RA={initial_coords.ra.deg}, DEC={initial_coords.dec.deg}")
        logging.debug(f"Subsequent Coordinates: RA={subsequent_coords.ra.deg}, DEC={subsequent_coords.dec.deg}")
        logging.debug(f"Observation Time: {times.iso}")
        
        orbit = calculate_orbit(initial_coords, subsequent_coords, times)
        logging.info(f"Calculated orbit: {orbit}")
        
        if orbit is not None:
            # Update database with orbit information
            insert_data('neossat_data.db', (filename, date_obs, exptime, brightness, str(orbit)))
            logging.debug("Orbit data updated in database successfully.")
            
            # Create interactive plot only if not in batch mode
            if not save_fig:
                logging.debug("Creating interactive plot.")
                create_interactive_plot(image_data, orbit)
                logging.debug("Interactive plot created successfully.")
        else:
            logging.warning("Orbit data is None. Skipping interactive visualization.")
        
        # Create two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        logging.debug("Displaying original image.")
        im1 = ax1.imshow(image_data, cmap=cmap, norm=norm)
        plt.colorbar(im1, ax=ax1, label='Flux')
        ax1.set_title(f'Original NEOSSAT Image: {os.path.basename(fits_file_path)}')
        ax1.set_xlabel('Pixel X')
        ax1.set_ylabel('Pixel Y')
        
        # Add some metadata from the header
        ax1.text(0.02, 0.98, f"Date: {date_obs}", transform=ax1.transAxes, va='top')
        ax1.text(0.02, 0.95, f"Exposure Time: {exptime} s", transform=ax1.transAxes, va='top')
        logging.debug("Added metadata annotations to the original image plot.")
        
        # Highlighted potential NEOs
        logging.debug("Generating highlighted image for potential NEOs.")
        highlighted_image = highlight_potential_neos(image_data, threshold_percentile)
        ax2.imshow(highlighted_image)
        ax2.set_title('Potential NEO Candidates')
        ax2.set_xlabel('Pixel X')
        ax2.set_ylabel('Pixel Y')
        
        plt.tight_layout()
        
        if save_fig and fig_path:
            plt.savefig(fig_path)
            plt.close()
            logging.debug(f"Figure saved to {fig_path}")
            # Update progress as success
            orbit_data = {"x": orbit.x.tolist(), "y": orbit.y.tolist()} if orbit else None
            # Progress updating should be handled externally during batch processing
        else:
            logging.info("Displaying plot.")
            plt.show()
            logging.info("Displaying the visualized FITS image with potential NEO highlights")
    except Exception as e:
        logging.error(f"An error occurred during visualization: {e}")
        # Update progress as failure
        # Progress updating should be handled externally during batch processing
        raise

def plot_histogram(fits_file_path, bins=100):
    logging.info(f"Plotting histogram for FITS file: {fits_file_path}")
    try:
        image_data, _ = load_fits_data(fits_file_path)
        logging.debug("Loaded image data for histogram plotting.")
        
        plt.figure(figsize=(10, 6))
        plt.hist(image_data.flatten(), bins=bins, log=True)
        plt.title(f'Histogram of Pixel Values: {os.path.basename(fits_file_path)}')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f"histogram_{os.path.basename(fits_file_path)}.png")
        plt.close()
        logging.info("Histogram plotted successfully and saved.")
    except Exception as e:
        logging.error(f"Error plotting histogram: {e}")
        raise

def process_all_fits_files(directory, output_video_path, frame_rate=1):
    logging.info(f"Starting processing of all FITS files in directory: {directory}")
    
    images_dir = os.path.join(directory, 'static', 'frames')
    os.makedirs(images_dir, exist_ok=True)
    logging.debug(f"Frames directory created at: {images_dir}")
    
    fits_files = sorted([f for f in os.listdir(directory) if f.endswith('.fits')])
    total_files = len(fits_files)
    logging.info(f"Found {total_files} FITS files in {directory}")
    
    for idx, fits_file in enumerate(fits_files, start=1):
        fits_path = os.path.join(directory, fits_file)
        logging.info(f"Processing file {idx}/{total_files}: {fits_file}")
        
        # Define the path to save the figure
        fig_path = os.path.join(images_dir, f'frame_{idx:04d}.png')
        
        try:
            # Visualize the FITS and save the plot as image
            visualize_fits(fits_path, save_fig=True, fig_path=fig_path)
            logging.debug(f"Saved visualization frame to {fig_path}")
            # Update progress as success
            image_data, _ = load_fits_data(fits_path)
            brightness = measure_brightness(image_data)
            update_progress(
                current=idx,
                total=total_files,
                filename=fits_file,
                status="Success",
                brightness=brightness,
                orbit=None  # Orbit details are already logged in the database
            )
        except Exception as e:
            logging.error(f"Failed to process {fits_file}: {e}")
            # Update progress as failure
            update_progress(
                current=idx,
                total=total_files,
                filename=fits_file,
                status=f"Failed: {str(e)}",
                brightness=None,
                orbit=None
            )
            continue  # Proceed to the next file
    
    # Create video from saved frames
    frame_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
    if not frame_files:
        logging.error("No frames found to create video.")
        return
    
    # Read first frame to get video properties
    first_frame_path = os.path.join(images_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        logging.error(f"Could not read image {first_frame_path}")
        return
    height, width, layers = first_frame.shape
    logging.debug(f"Video frame dimensions: width={width}, height={height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    logging.debug(f"VideoWriter initialized with path: {output_video_path}, frame rate: {frame_rate}, frame size: ({width}, {height})")
    
    for frame_file in frame_files:
        frame_path = os.path.join(images_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            logging.warning(f"Could not read image {frame_path}, skipping.")
            continue
        video.write(frame)
        logging.debug(f"Added frame to video: {frame_path}")
    
    video.release()
    logging.info(f"Video created successfully at {output_video_path}")
    
    # Optionally, clean up frames directory
    # cleanup_frames(directory)

def cleanup_frames(directory):
    frames_dir = os.path.join(directory, 'static', 'frames')
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
        logging.debug(f"Frames directory '{frames_dir}' has been deleted.")

if __name__ == "__main__":
    # Initialize the database
    logging.debug("Initializing the database.")
    create_database('neossat_data.db')
    logging.debug("Database initialized successfully.")
    
    fits_directory = "D:/AI Nasa/NEOSSAT_Data"
    output_video_path = os.path.join(fits_directory, "visualization_video.mp4")
    
    try:
        logging.info("Starting batch visualization and video creation process")
        process_all_fits_files(fits_directory, output_video_path, frame_rate=1)
        logging.info("Batch visualization and video creation completed successfully")
    except Exception as e:
        logging.exception(f"An error occurred during the batch visualization process: {e}")
        print(f"An error occurred: {e}")
        print("Please ensure all FITS files are valid and the output directory is writable.")
        print(f"Check the log file '{log_file}' for detailed debug information.")