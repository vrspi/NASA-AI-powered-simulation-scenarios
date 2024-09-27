import multiprocessing

def process_file(file_path):
    # Processing logic
    pass

if __name__ == "__main__":
    files = ["file1.fits", "file2.fits", ...]
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(process_file, files)