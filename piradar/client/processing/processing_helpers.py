"""
Common helper functions for radar processing.

"""

import logging
import pickle
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy import signal as sp_signal


def apply_window_2d(data: np.ndarray, window_type: str) -> np.ndarray:
    """Apply separable window across chirps and samples.

    Supports inputs shaped as (num_chirps, num_samples) or
    (num_chirps, num_samples, num_channels).
    """
    if window_type == "none":
        return data

    if data.ndim == 3:
        num_chirps, num_samples, _num_channels = data.shape
    elif data.ndim == 2:
        num_chirps, num_samples = data.shape
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")

    if window_type == "hamming":
        range_window = sp_signal.windows.hamming(num_samples)
        doppler_window = sp_signal.windows.hamming(num_chirps)
    elif window_type == "hanning":
        range_window = sp_signal.windows.hann(num_samples)
        doppler_window = sp_signal.windows.hann(num_chirps)
    elif window_type == "blackman":
        range_window = sp_signal.windows.blackman(num_samples)
        doppler_window = sp_signal.windows.blackman(num_chirps)
    else:
        raise ValueError(f"Unknown window type: {window_type}")

    if data.ndim == 3:
        # Broadcast windows across channels
        windowed = data * range_window[np.newaxis, :, np.newaxis]
        windowed = windowed * doppler_window[:, np.newaxis, np.newaxis]
    else:  # 2D
        windowed = data * range_window[np.newaxis, :]
        windowed = windowed * doppler_window[:, np.newaxis]

    return windowed


def compute_range_doppler_fft(windowed_data: np.ndarray, range_bins: Optional[int] = None,
                             doppler_bins: Optional[int] = None) -> np.ndarray:
    """Compute 2D FFT for range-Doppler processing.

    Accepts input of shape (num_chirps, num_samples) or
    (num_chirps, num_samples, num_channels). The FFT is applied along
    chirp (Doppler) and sample (range) axes. If channels are present,
    magnitudes are summed across channels.
    """
    if windowed_data.ndim == 3:
        num_chirps, num_samples, _num_channels = windowed_data.shape
    elif windowed_data.ndim == 2:
        num_chirps, num_samples = windowed_data.shape
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {windowed_data.shape}")

    # Use provided bin counts or default to data dimensions
    if range_bins is None:
        range_bins = num_samples
    if doppler_bins is None:
        doppler_bins = num_chirps

    # Range FFT (along samples axis)
    range_fft = np.fft.fft(windowed_data, n=range_bins, axis=1)

    # Doppler FFT (along chirps axis)
    doppler_fft = np.fft.fft(range_fft, n=doppler_bins, axis=0)

    # Shift to center DC components
    range_doppler = np.fft.fftshift(doppler_fft, axes=(0, 1))

    # Magnitude and channel combination (if any)
    magnitude = np.abs(range_doppler)
    if magnitude.ndim == 3:
        magnitude = np.sum(magnitude, axis=2)

    return magnitude


def calculate_range_axis_from_config(num_range_bins: int, sample_rate_hz: float, slope_hz_per_s: float) -> np.ndarray:
    """Calculate positive range axis (cm) using sample rate and chirp slope from config.

    Uses R = (c / (2*k)) * f_b, with f_b from FFT frequency bins of ADC samples.
    Returns only the positive half after FFT shift (length = num_range_bins//2).
    """
    c = 3e8  # m/s
    if not (isinstance(sample_rate_hz, (int, float)) and sample_rate_hz > 0.0):
        raise ValueError("Invalid sample_rate_hz for range axis")
    if not (isinstance(slope_hz_per_s, (int, float)) and slope_hz_per_s > 0.0):
        raise ValueError("Invalid slope_hz_per_s for range axis")
    freq = np.fft.fftfreq(num_range_bins, d=1.0 / float(sample_rate_hz))
    freq_shifted = np.fft.fftshift(freq)
    positive_freqs = freq_shifted[num_range_bins // 2 :]
    range_m = (c / (2.0 * float(slope_hz_per_s))) * positive_freqs
    return range_m * 100.0  # to cm


def calculate_velocity_axis_from_config(num_doppler_bins: int, frame_length: int, frame_duration_s: float,
                                        center_freq_hz: float, chirp_repetition_time_s: Optional[float] = None) -> np.ndarray:
    """Calculate full velocity axis (m/s) using slow-time sampling from frame config.

    v = (c / (2*fc)) * f_d where f_d are FFT frequency bins across chirps and
    slow-time sampling rate f_slow = 1 / PRI with PRI = chirp_repetition_time_s (chirp-to-chirp period).
    
    The unambiguous velocity range is ±λ/(4*T_c) where T_c is the chirp repetition time.
    At 64 GHz with T_c = 100µs, this gives ±11.7 m/s.
    
    Args:
        num_doppler_bins: Number of Doppler bins (FFT size along chirp axis)
        frame_length: Number of chirps per frame (for backward compatibility, not used in calculation)
        frame_duration_s: Total frame duration in seconds (for backward compatibility, not used in calculation)
        center_freq_hz: Center frequency in Hz
        chirp_repetition_time_s: Chirp repetition time (PRI) in seconds. Defaults to 100µs if not provided.
    
    Returns:
        Full velocity axis (length = num_doppler_bins), centered around 0, in m/s.
    """
    c = 3e8
    if not (isinstance(center_freq_hz, (int, float)) and center_freq_hz > 0.0):
        raise ValueError("Invalid center_freq_hz for velocity axis")
    
    # Use chirp repetition time directly (default 100µs)
    if chirp_repetition_time_s is None:
        pri_s = 100e-6  # Default: 100 microseconds
    else:
        if not (isinstance(chirp_repetition_time_s, (int, float)) and chirp_repetition_time_s > 0.0):
            raise ValueError("Invalid chirp_repetition_time_s for velocity axis")
        pri_s = float(chirp_repetition_time_s)
    
    logging.debug(f"PRI (chirp repetition time): {pri_s*1e6:.2f} µs")
    
    # Calculate Doppler frequency bins
    # Sampling rate for slow-time (across chirps) is 1/PRI
    freq = np.fft.fftfreq(num_doppler_bins, d=pri_s)
    freq_shifted = np.fft.fftshift(freq)
    
    # Convert Doppler frequency to velocity: v = (c/(2*fc)) * f_d
    velocity = (c / (2.0 * float(center_freq_hz))) * freq_shifted
    
    # Calculate unambiguous velocity range for logging
    wavelength = c / float(center_freq_hz)
    v_max_unambiguous = wavelength / (4.0 * pri_s)
    logging.debug(f"Velocity axis range: [{velocity[0]:.2f}, {velocity[-1]:.2f}] m/s")
    logging.debug(f"Unambiguous velocity range: ±{v_max_unambiguous:.2f} m/s (±{v_max_unambiguous*3.6:.1f} km/h)")
    
    return velocity


def process_range_doppler(data: np.ndarray, window_type: str, range_bins: Optional[int], 
                         doppler_bins: Optional[int], db_min: float, db_max: float) -> np.ndarray:
    """Complete range-Doppler processing pipeline.
    
    Args:
        data: Input radar data
        window_type: Window function type
        range_bins: Optional number of range bins
        doppler_bins: Optional number of doppler bins
        db_min: Minimum dynamic range in dB for display
        db_max: Maximum dynamic range in dB for display
    
    Returns:
        Log-magnitude range-Doppler spectrum
    """
    # 1. Apply 2D window
    windowed = apply_window_2d(data, window_type)
    
    # 2. Compute 2D FFT
    rd_spectrum = compute_range_doppler_fft(windowed, range_bins, doppler_bins)
    
    # 3. Convert to log-magnitude with dynamic range
    db_range = db_max - db_min
    log_mag = to_log_magnitude(rd_spectrum, db_range)
    
    return log_mag


def extract_time_series(frame: np.ndarray, chirp_index: int) -> Optional[np.ndarray]:
    """Return array with shape (N, C) where C>=3, containing time-domain samples.

    Args:
        frame: The frame to extract the time-series from of shape (num_chirps, num_samples, num_rx).
        chirp_index: The chirp index to extract or -1 for the mean of all chirps.
    
    Returns:
        The time-series array with shape (num_samples, num_rx).
    """

    if chirp_index < 0:
        return frame.mean(axis=0)
    else:
        return frame[chirp_index]



def apply_window_1d(signal: np.ndarray, window_type: str) -> np.ndarray:
    """Apply window function to a 1D signal (time-domain samples).
    
    Args:
        signal: Input signal of shape (num_samples, num_channels)
        window_type: Window function type ('hamming', 'hanning', 'blackman', or 'none')
    
    Returns:
        Windowed signal with same shape as input
    """
    if window_type == "none":
        return signal
    
    N = signal.shape[0]
    
    if window_type == "hamming":
        window = sp_signal.windows.hamming(N)
    elif window_type == "hanning":
        window = sp_signal.windows.hann(N)
    elif window_type == "blackman":
        window = sp_signal.windows.blackman(N)
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    
    # Apply window to each channel
    windowed = signal * window[:, np.newaxis]
    return windowed


def compute_fft_spectrum(windowed_signal: np.ndarray) -> np.ndarray:
    """Compute FFT spectrum for each channel.
    
    Args:
        windowed_signal: Input signal of shape (num_samples, num_channels)
    
    Returns:
        FFT magnitude spectrum of shape (num_samples, num_channels)
    """
    # Compute FFT for each channel
    fft_result = np.fft.fft(windowed_signal, axis=0)
    
    # Take magnitude and shift to center DC
    return np.fft.fftshift(fft_result, axes=0)
    
    

def calculate_frequency_axis(num_bins: int, sample_rate: float) -> np.ndarray:
    """Calculate frequency axis in Hz from frequency bins.
    
    Args:
        num_bins: Number of frequency bins
        sample_rate: Sample rate in Hz
        
    Returns:
        Array of frequency values in Hz (centered around 0)
    """
    # Frequency resolution
    freq_resolution = sample_rate / num_bins
    
    # Create frequency axis centered around 0 (like fftshift)
    freq_axis = np.arange(num_bins) * freq_resolution
    freq_axis = freq_axis - freq_axis[num_bins // 2]  # Center around 0
    
    return freq_axis


def to_log_magnitude(spectrum: np.ndarray, db_range: float) -> np.ndarray:
    """Convert to log-magnitude spectrum with specified dynamic range.
    
    Args:
        spectrum: Input spectrum (magnitude)
        db_range: Dynamic range in dB
    
    Returns:
        Log-magnitude spectrum clipped to the specified dynamic range
    """
    # Avoid log(0) by adding small epsilon
    eps = 1e-12
    log_mag = 20 * np.log10(spectrum + eps)
    
    # Normalize to specified dynamic range
    max_val = np.max(log_mag)
    min_val = max_val - db_range
    log_mag = np.clip(log_mag, min_val, max_val)
    
    return log_mag


def non_coherent_integration(spectrum: np.ndarray, method: str = "sum") -> np.ndarray:
    """Perform non-coherent integration across channels.
    
    Args:
        spectrum: Complex or magnitude spectrum of shape (num_bins, num_channels)
        method: Integration method ('sum', 'mean', or 'max')
    
    Returns:
        Integrated spectrum as 1D array
    """
    # Take magnitude if complex
    if np.iscomplexobj(spectrum):
        magnitude = np.abs(spectrum)
    else:
        magnitude = spectrum
    
    if method == "sum":
        return np.sum(magnitude, axis=1)
    elif method == "mean":
        return np.mean(magnitude, axis=1)
    elif method == "max":
        return np.max(magnitude, axis=1)
    else:
        raise ValueError(f"Unknown NCI method: {method}. Use 'sum', 'mean', or 'max'.")


def process_frequency_domain(ts: np.ndarray, window_type: str, nci_method: str, db_range: float) -> np.ndarray:
    """Complete frequency domain processing pipeline.
    
    Args:
        ts: Time-domain signal of shape (num_samples, num_channels)
        window_type: Window function type
        nci_method: NCI method ('sum', 'mean', or 'max')
        db_range: Dynamic range in dB
    
    Returns:
        Log-magnitude spectrum as 1D array
    """
    # 1. Apply window
    windowed = apply_window_1d(ts, window_type)
    
    # 2. Compute FFT spectrum
    spectrum = compute_fft_spectrum(windowed)
    
    # 3. Non-Coherent Integration
    nci_spectrum = non_coherent_integration(spectrum, nci_method)
    
    # 4. Convert to log-magnitude
    log_mag = to_log_magnitude(nci_spectrum, db_range)
    
    return log_mag

