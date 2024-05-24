"""
Script to interpolate the NaN found in the result. At first we thought of using this approach if nnan were a few. Then,
we discovered the real mistake.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from test import spectrogram_to_audio


def load_data(file_path):
    """Load spectrogram data from a numpy file."""
    return np.load(file_path)


def check_nan_values(data):
    """Check for NaN values in the data array."""
    nan_mask = np.isnan(data)
    return nan_mask, np.any(nan_mask)


def prepare_interpolator(data, nan_mask):
    """Prepare the RegularGridInterpolator for NaN values."""
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])
    data_filled = np.nan_to_num(data, nan=0.0)
    interpolator = RegularGridInterpolator(
        (x, y, z), data_filled, method="linear", bounds_error=False, fill_value=None
    )
    points_to_interpolate = np.array(np.where(nan_mask)).T
    return interpolator, points_to_interpolate


def interpolate_nan_values(data, nan_mask):
    """Interpolate NaN values in the data array using a linear method."""
    interpolator, points_to_interpolate = prepare_interpolator(data, nan_mask)
    interpolated_values = interpolator(points_to_interpolate)
    data[np.where(nan_mask)] = interpolated_values


def save_data(file_path, data):
    """Save the modified data back to a numpy file."""
    np.save(file_path, data)


def main():
    """Load, check, interpolate NaNs, and save spectrogram data."""
    file_path = "test_results/generated_npy/generated_spectrogram.npy"
    data = load_data("test_results/generated_npy/generated_spectrogram.npy")
    nan_mask, has_nan = check_nan_values(data)
    if not has_nan:
        print("No values to interpolate.")
    else:
        interpolate_nan_values(data, nan_mask)
    save_data(file_path, data)
    spectrogram_to_audio(data)


if __name__ == "__main__":
    main()
