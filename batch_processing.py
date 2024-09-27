import os
from astropy.io import fits

def process_fits_file(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[0].data
        # Apply preprocessing and analysis
        # ...
    return results

def batch_process(directory):
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.fits'):
            file_path = os.path.join(directory, filename)
            result = process_fits_file(file_path)
            results.append(result)
    return results

if __name__ == "__main__":
    data_directory = "D:/AI Nasa/NEOSSAT_Data"
    all_results = batch_process(data_directory)
    # Further processing of results