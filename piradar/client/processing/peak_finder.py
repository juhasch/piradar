"""
Peak finding algorithms for radar range data.

This module provides peak detection algorithms specifically designed for radar
range-amplitude data, including center of gravity peak detection and highest
peak detection with self-coupling rejection.
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


def center_of_gravity_peak_detection(complex_data: np.ndarray, 
                                   window_size: int = 3, 
                                   threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in complex range-amplitude data using center of gravity method.
    
    This function provides sub-bin accuracy by calculating the center of gravity
    around detected peaks, which is more accurate than simple peak finding.
    
    Parameters:
    -----------
    complex_data : np.ndarray
        Complex array with range-amplitude data (typically from FFT)
    window_size : int, optional
        Size of window around peak for center of gravity calculation (default: 3)
    threshold : float, optional
        Optional threshold for initial peak detection (e.g., minimum amplitude)
    
    Returns:
    --------
    peak_positions : np.ndarray
        Array of precise peak positions (float indices)
    peak_amplitudes : np.ndarray
        Array of amplitudes at peak positions
    """
    # Calculate amplitudes (magnitude of complex data)
    amplitudes = np.abs(complex_data)
    
    # Initial peak detection with scipy
    peaks, properties = find_peaks(amplitudes, height=threshold)
    
    # Lists for results
    peak_positions = []
    peak_amplitudes = []
    
    # Half window size (symmetric around peak)
    half_window = window_size // 2
    
    for peak_idx in peaks:
        # Define window around the peak
        start_idx = max(0, peak_idx - half_window)
        end_idx = min(len(complex_data), peak_idx + half_window + 1)
        
        # Extract amplitudes and indices in window
        window_amplitudes = amplitudes[start_idx:end_idx]
        window_indices = np.arange(start_idx, end_idx)
        
        # Calculate center of gravity (weighted average of indices)
        if np.sum(window_amplitudes) > 0:  # Avoid division by zero
            cog_position = np.sum(window_indices * window_amplitudes) / np.sum(window_amplitudes)
        else:
            cog_position = peak_idx  # Fallback if amplitudes are zero
            
        # Use the actual peak amplitude (not interpolated) for more accurate representation
        # The center of gravity gives us sub-bin position accuracy, but the peak amplitude
        # should be the actual highest value found, not an interpolated value
        peak_amplitude = amplitudes[peak_idx]
        
        peak_positions.append(cog_position)
        peak_amplitudes.append(peak_amplitude)
    
    return np.array(peak_positions), np.array(peak_amplitudes)


def parabolic_peak_detection(complex_data: np.ndarray, 
                           window_size: int = 3, 
                           threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find peaks using parabolic interpolation for sub-bin accuracy.
    
    This method fits a parabola to the peak and its neighbors to find the
    true peak position and amplitude, which is often more accurate than
    center of gravity for symmetric peaks.
    
    Parameters:
    -----------
    complex_data : np.ndarray
        Complex array with range-amplitude data (typically from FFT)
    window_size : int, optional
        Size of window around peak (must be 3 for parabolic fit)
    threshold : float, optional
        Optional threshold for initial peak detection
    
    Returns:
    --------
    peak_positions : np.ndarray
        Array of precise peak positions (float indices)
    peak_amplitudes : np.ndarray
        Array of interpolated peak amplitudes
    """
    # Calculate amplitudes (magnitude of complex data)
    amplitudes = np.abs(complex_data)
    
    # Initial peak detection with scipy
    peaks, properties = find_peaks(amplitudes, height=threshold)
    
    # Lists for results
    peak_positions = []
    peak_amplitudes = []
    
    for peak_idx in peaks:
        # Ensure we have at least 3 points for parabolic fit
        if peak_idx > 0 and peak_idx < len(amplitudes) - 1:
            # Get three points: left, peak, right
            y1 = amplitudes[peak_idx - 1]
            y2 = amplitudes[peak_idx]
            y3 = amplitudes[peak_idx + 1]
            
            # Parabolic interpolation formula
            # Peak position offset from center point
            offset = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
            
            # Peak position (can be fractional)
            peak_position = peak_idx + offset
            
            # Peak amplitude using parabolic interpolation
            peak_amplitude = y2 - 0.25 * (y1 - y3) * offset
            
            peak_positions.append(peak_position)
            peak_amplitudes.append(peak_amplitude)
        else:
            # Fallback to discrete peak if we can't do parabolic fit
            peak_positions.append(float(peak_idx))
            peak_amplitudes.append(amplitudes[peak_idx])
    
    return np.array(peak_positions), np.array(peak_amplitudes)


def find_highest_peak(complex_data: np.ndarray,
                     range_axis: Optional[np.ndarray] = None,
                     ignore_range_0: bool = True,
                     min_range_cm: float = 5.0,
                     max_range_cm: Optional[float] = None,
                     window_size: int = 3,
                     threshold: Optional[float] = None,
                     method: str = 'parabolic') -> dict:
    """
    Find the highest peak in radar range data, optionally ignoring range 0.
    
    This function is specifically designed for radar applications where
    self-coupling at range 0 needs to be ignored.
    
    Parameters:
    -----------
    complex_data : np.ndarray
        Complex range-amplitude data (typically from FFT)
    range_axis : np.ndarray, optional
        Range axis in cm. If provided, returns actual range values
    ignore_range_0 : bool, default True
        Whether to ignore peaks at range 0 (self-coupling)
    min_range_cm : float, default 5.0
        Minimum range to consider (in cm)
    max_range_cm : float, optional
        Maximum range to consider (in cm)
    window_size : int, default 3
        Window size for peak detection
    threshold : float, optional
        Minimum amplitude threshold for peak detection
    method : str, default 'parabolic'
        Peak detection method: 'parabolic' or 'centroid'
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'peak_position': Peak position (index or range)
        - 'peak_amplitude': Peak amplitude
        - 'peak_range_cm': Peak range in cm (if range_axis provided)
        - 'all_peaks': All detected peaks
    """
    # Calculate amplitudes
    amplitudes = np.abs(complex_data)
    
    # Create mask to exclude unwanted ranges
    mask = np.ones(len(amplitudes), dtype=bool)
    
    if ignore_range_0:
        # Exclude first few bins (typically range 0)
        mask[:max(1, len(amplitudes) // 20)] = False  # Exclude first 5% of bins
    
    if range_axis is not None:
        if min_range_cm is not None:
            mask &= (range_axis >= min_range_cm)
        if max_range_cm is not None:
            mask &= (range_axis <= max_range_cm)
    
    # Apply mask to data
    masked_amplitudes = amplitudes.copy()
    masked_amplitudes[~mask] = 0
    
    # Find all peaks in the valid range using specified method
    if method == 'parabolic':
        peak_positions, peak_amplitudes = parabolic_peak_detection(
            complex_data, window_size, threshold
        )
    else:  # default to centroid
        peak_positions, peak_amplitudes = center_of_gravity_peak_detection(
            complex_data, window_size, threshold
        )
    
    # Filter peaks based on mask
    valid_peaks = []
    valid_amplitudes = []
    
    for pos, amp in zip(peak_positions, peak_amplitudes):
        pos_idx = int(round(pos))
        if pos_idx < len(mask) and mask[pos_idx]:
            valid_peaks.append(pos)
            valid_amplitudes.append(amp)
    
    if len(valid_peaks) == 0:
        logger.warning("No valid peaks found in the specified range")
        return {
            'peak_position': None,
            'peak_amplitude': None,
            'peak_range_cm': None,
            'all_peaks': {'positions': [], 'amplitudes': []}
        }
    
    # Find highest peak
    max_amp_idx = np.argmax(valid_amplitudes)
    highest_peak_pos = valid_peaks[max_amp_idx]
    highest_peak_amp = valid_amplitudes[max_amp_idx]
    
    # Calculate range if range_axis is provided
    peak_range_cm = None
    if range_axis is not None:
        peak_range_cm = np.interp(highest_peak_pos, np.arange(len(range_axis)), range_axis)
    
    result = {
        'peak_position': highest_peak_pos,
        'peak_amplitude': highest_peak_amp,
        'peak_range_cm': peak_range_cm,
        'all_peaks': {
            'positions': valid_peaks,
            'amplitudes': valid_amplitudes
        }
    }
    
    if peak_range_cm is not None:
      logger.debug(f"Highest peak found at position {highest_peak_pos:.2f}, "
                f"amplitude {highest_peak_amp:.2f} dB")
      logger.debug(f"Range: {peak_range_cm:.1f} cm")
    
    return result


def find_peaks_in_range(complex_data: np.ndarray,
                        range_axis: np.ndarray,
                        target_range_cm: float,
                        tolerance_cm: float = 10.0,
                        window_size: int = 3,
                        threshold: Optional[float] = None) -> dict:
    """
    Find peaks within a specific range window around a target distance.
    
    This is useful for tracking specific targets or objects at known distances.
    
    Parameters:
    -----------
    complex_data : np.ndarray
        Complex range-amplitude data
    range_axis : np.ndarray
        Range axis in cm
    target_range_cm : float
        Target range to search around
    tolerance_cm : float, default 10.0
        Tolerance window around target range
    window_size : int, default 3
        Window size for center of gravity calculation
    threshold : float, optional
        Minimum amplitude threshold
    
    Returns:
    --------
    dict
        Dictionary containing peak information within the specified range
    """
    # Create range mask
    range_mask = (range_axis >= (target_range_cm - tolerance_cm)) & \
                 (range_axis <= (target_range_cm + tolerance_cm))
    
    if not np.any(range_mask):
        logger.warning(f"No data points found in range {target_range_cm}±{tolerance_cm} cm")
        return {
            'peak_position': None,
            'peak_amplitude': None,
            'peak_range_cm': None,
            'found': False
        }
    
    # Extract data in range
    range_indices = np.where(range_mask)[0]
    range_data = complex_data[range_mask]
    range_axis_subset = range_axis[range_mask]
    
    # Find peaks in the subset
    peak_positions, peak_amplitudes = center_of_gravity_peak_detection(
        range_data, window_size, threshold
    )
    
    if len(peak_positions) == 0:
        return {
            'peak_position': None,
            'peak_amplitude': None,
            'peak_range_cm': None,
            'found': False
        }
    
    # Find highest peak in range
    max_amp_idx = np.argmax(peak_amplitudes)
    peak_pos = peak_positions[max_amp_idx]
    peak_amp = peak_amplitudes[max_amp_idx]
    
    # Convert back to full array index
    full_index = range_indices[0] + peak_pos
    
    # Calculate actual range
    peak_range_cm = np.interp(peak_pos, np.arange(len(range_axis_subset)), range_axis_subset)
    
    return {
        'peak_position': full_index,
        'peak_amplitude': peak_amp,
        'peak_range_cm': peak_range_cm,
        'found': True
    }


# Example usage and testing
if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    n_samples = 100
    t = np.arange(n_samples)
    
    # Create synthetic radar data with multiple peaks
    signal = np.zeros(n_samples, dtype=complex)
    
    # Add peaks at different ranges
    peak1_pos = 20
    peak1_amp = 2.0
    signal += peak1_amp * np.exp(1j * 2 * np.pi * t / 20) * np.exp(-(t - peak1_pos)**2 / 10)
    
    peak2_pos = 60
    peak2_amp = 1.5
    signal += peak2_amp * np.exp(1j * 2 * np.pi * t / 15) * np.exp(-(t - peak2_pos)**2 / 8)
    
    # Add noise
    noise = 0.1 * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
    complex_data = signal + noise
    
    # Create range axis (example: 0-200 cm)
    range_axis = np.linspace(0, 200, n_samples)
    
    # Test center of gravity peak detection
    print("Testing center of gravity peak detection:")
    peak_positions, peak_amplitudes = center_of_gravity_peak_detection(
        complex_data, window_size=5, threshold=0.2
    )
    print(f"Peak positions: {peak_positions}")
    print(f"Peak amplitudes: {peak_amplitudes}")
    
    # Test highest peak detection
    print("\nTesting highest peak detection:")
    result = find_highest_peak(complex_data, range_axis, ignore_range_0=True)
    print(f"Highest peak: {result}")
    
    # Test range-specific peak detection
    print("\nTesting range-specific peak detection:")
    range_result = find_peaks_in_range(
        complex_data, range_axis, target_range_cm=60, tolerance_cm=20
    )
    print(f"Peak in range: {range_result}")
