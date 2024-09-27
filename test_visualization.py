import pytest
from visualize import load_fits_data

def test_load_fits_data():
    file_path = "path_to_valid_fits_file.fits"
    data, header = load_fits_data(file_path)
    assert data is not None
    assert header is not None

def test_load_fits_data_invalid():
    file_path = "invalid_path.fits"
    with pytest.raises(FileNotFoundError):
        load_fits_data(file_path)