# 01 - Extracting the data and visualization
# ***
#
# **Tutorial :** This tutorial provides a step to step guide to extract data from the the Canadian astronomy data centre (CADC) and the open data portal of the Canadian Space Agency. 
# **Mission and Instrument :** NEOSSAT  
# **Astronomical Target :** Detecting and tracking near earth objects      
# **System Requirements :** Python 3.9 or later  
# **Tutorial Level :** Basic  
#
# For more information on on the NEOSSAT space telescope and the FITS files please consult NEOSSat FITS Image User's Guide via the following link: https://donnees-data.asc-csa.gc.ca/users/OpenData_DonneesOuvertes/pub/NEOSSAT/Supporting%20Documents/CSA-NEOSSAT-MAN-0002_FITS_IMAGE_UGUIDE-v4-00.pdf
#
# **Extracting data from the open data portal of the Canadian Space Agency (CSA):**: The Canadian Space Agency has a dedicated page to NEOSSAT datasets on its open data and information webpage which can be found via the following link: https://donnees-data.asc-csa.gc.ca/en/dataset/9ae3e718-8b6d-40b7-8aa4-858f00e84b30
#
# **Extracting data from the Canadian Astronomy Data Centre (CADC)**: CADC recommends the installation of CADCdata package for usage in python. Documentation on how to access the library is available with the pydoc cadc. For more information visit: https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/doc/data/#library

# ## Option 1 - Extracting NEOSSAT data from the Canadian Space Agency (CSA)
# With the Canadian Space Agency dataset you can filter the NEOSSAT data based on the year, day of year and time.
#
# To understand the structure of data: 
#
# The folders show the Year. The subfolders show the day of year. The .fits files are named according to the Year, Day of Year and Hour
#
# Ex: NEOS_SCI_2021002000257.fits => [Year: 2021], [Day of Year: 002], [Time: 00:02:57]

# ### Importing required libraries


from astropy.visualization import astropy_mpl_style
from astropy.io import fits
import matplotlib.pyplot as plt
from ftplib import FTP
import os
import re
from astropy.coordinates import SkyCoord
import astropy.units as u

# The function below will filter through the CSA NEOSSAT data based on the year and the day of year. It will then download the .fits files to the local path. 
def download_fits_files(year, day_of_year, local_directory):
    ftp_host = 'ftp.asc-csa.gc.ca'
    ftp = FTP(ftp_host)
    ftp.login()
    
    ftp_directory = f'/users/OpenData_DonneesOuvertes/pub/NEOSSAT/ASTRO/{year}/{day_of_year}/'
    print(f"Attempting to access FTP directory: {ftp_directory}")
    try:
        ftp.cwd(ftp_directory)
    except Exception as e:
        print(f"Error accessing FTP directory: {e}")
        return
    
    # List all .fits files in the directory
    try:
        fits_files = [file for file in ftp.nlst() if file.endswith('.fits')]
        print(f"Found {len(fits_files)} .fits files")
    except Exception as e:
        print(f"Error listing files: {e}")
        return
    
    # Ensure the local directory exists
    print(f"Attempting to create local directory: {local_directory}")
    try:
        os.makedirs(local_directory, exist_ok=True)
        print(f"Local directory created/verified: {local_directory}")
    except Exception as e:
        print(f"Error creating local directory: {e}")
        return
    
    for fits_file in fits_files:
        local_filename = os.path.join(local_directory, fits_file)
        print(f"Attempting to download: {fits_file}")
        print(f"Saving to: {local_filename}")
        try:
            with open(local_filename, 'wb') as f:
                ftp.retrbinary('RETR ' + fits_file, f.write)
            
            # Verify that the file was created and has content
            if os.path.exists(local_filename) and os.path.getsize(local_filename) > 0:
                print(f"Successfully downloaded and verified: {fits_file}")
            else:
                print(f"File was not created or is empty: {local_filename}")
        except Exception as e:
            print(f"Error downloading {fits_file}: {e}")
    
    ftp.quit()
    print("Download process completed.")
    
    # List all files in the directory after download
    print(f"\nListing files in {local_directory}:")
    for file in os.listdir(local_directory):
        print(file)

# Select the year, and the day of year you are interested in. In this example we are downloading the data from Feb 19th, 2024
if __name__ == "__main__":
    year = '2024'
    day_of_year = '050'
    local_directory = r'D:/AI Nasa/NEOSSAT_Data'  # Using raw string for Windows path
    print(f"Starting download process for year: {year}, day: {day_of_year}")
    print(f"Local directory set to: {local_directory}")
    download_fits_files(year, day_of_year, local_directory)

# Another way to filter through the data by selecting the time and only download the range of .fits file that matches the time criteria. We start by defining the function: 
def download_fits_files_in_time_range(year, day_of_year, start_time, end_time, local_directory):
    ftp_host = 'ftp.asc-csa.gc.ca'
    ftp = FTP(ftp_host)
    ftp.login()
    
    ftp_directory = f'/users/OpenData_DonneesOuvertes/pub/NEOSSAT/ASTRO/{year}/{day_of_year}/'
    ftp.cwd(ftp_directory)
    
    # List all .fits files in the directory
    fits_files = [file for file in ftp.nlst() if file.endswith('.fits')]
    
    start_time_str = start_time.zfill(6)
    end_time_str = end_time.zfill(6)
    
    filtered_files = []
    for fits_file in fits_files:
        match = re.search(r'(\d{6})\.fits$', fits_file)
        if match:
            file_time_str = match.group(1)
            if start_time_str <= file_time_str <= end_time_str:
                filtered_files.append(fits_file)
    
    # Ensure the local directory exists
    os.makedirs(local_directory, exist_ok=True)
    
    if filtered_files:
        for fits_file in filtered_files:
            local_filename = os.path.join(local_directory, fits_file)
            with open(local_filename, 'wb') as f:
                ftp.retrbinary('RETR ' + fits_file, f.write)
        print("Download complete.")
        return filtered_files
    else:
        print("No files found in the specified time range.")
        return None

# Select the year, the day of year and the time range you are interested in. In this example we are downloading the data from Feb 19th, 2024 from 00:02:57 to 01:09:20
def main_time_range():
    year = '2024'
    day_of_year = '002'
    start_time = '000000'  # If the time in mind is 00:00:00 write 000000 to match the last 6 digits filename of the .fits file
    end_time = '010257'    # If the time in mind is 01:02:57 write 010257 to match the last 6 digits filename of the .fits file
    local_directory = 'D:/AI Nasa/NEOSSAT_Data/Part 1/2024/time_range'
    
    downloaded_files = download_fits_files_in_time_range(year, day_of_year, start_time, end_time, local_directory)

if __name__ == "__main__":
    main_time_range()

# Let's visualize the selected range of the .fits files. 

# Initialize the CADC client 
# Note: Ensure that the Cadc class or module is properly imported or defined
from cadc import Cadc  # You may need to install and import the appropriate CADC library

cadc = Cadc()

# Print information about available collections 
for collection, details in sorted(cadc.get_collections().items()):
    print(f'{collection} : {details}')

# Define the target coordinates and search radius based on your preference
coords = SkyCoord(240, -30, unit='deg')  # RA, DEC, Unit
radius = 2 * u.deg

# Query CADC for data within the specified region and collection 
results = cadc.query_region(coords, radius, collection='NEOSSAT')
print(results)

# Filter the query results to select data with a specific 'time_exposure'. 
# In this example 'time_exposure' greater than 50 is selected

filtered_results = results[results['time_exposure'] > 50.0]

# Access data points from the filtered results 
print(filtered_results['time_exposure'][100])
print(filtered_results['position_dimension_naxis2'][100])
print(filtered_results['position_dimension_naxis1'][100])
print(filtered_results['instrument_keywords'][100])
print(filtered_results['metaRelease'][100])

# Get a list of image URLs based on the filtered results
image_list = cadc.get_image_list(filtered_results, coords, radius)

# Print the number of images in the image_list 
print(len(image_list))

# Print the last URL in the image_list 
print(image_list[-1])

# Get the filename of the 100th image from the image_list 
filename = image_list[100]
print(filename)

# Let's visualize the selected .fits file. 

# Read the FITS image data from the file
image_data = fits.getdata(filename, ext=0)

# Show the image file
plt.style.use(astropy_mpl_style)
plt.figure()
plt.axis('off')
plt.imshow(image_data, cmap='gray')
plt.show()