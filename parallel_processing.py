import multiprocessing
from visualize import visualize_fits

def parallel_process_fits_files(files):
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(visualize_fits, files)