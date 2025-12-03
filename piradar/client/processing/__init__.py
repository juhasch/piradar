"""
Signal processing utilities for PiRadar Client.

Provides functions for radar signal processing including windowing,
FFT operations, and range-Doppler processing.
"""

from .processing_helpers import (
    apply_window_2d,
    compute_range_doppler_fft,
    calculate_range_axis_from_config,
    calculate_velocity_axis_from_config,
    process_range_doppler,
    extract_time_series,
    apply_window_1d,
    compute_fft_spectrum,
    calculate_frequency_axis,
    process_frequency_domain,
    to_log_magnitude,
    non_coherent_integration,
)
from .multi_radar import MultiRadarCoordinator

__all__ = [
    "apply_window_2d",
    "compute_range_doppler_fft",
    "calculate_range_axis_from_config",
    "calculate_velocity_axis_from_config",
    "process_range_doppler",
    "extract_time_series",
    "apply_window_1d",
    "compute_fft_spectrum",
    "calculate_frequency_axis",
    "process_frequency_domain",
    "to_log_magnitude",
    "non_coherent_integration",
    "MultiRadarCoordinator",
]
