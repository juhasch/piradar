"""
Helper functions for radar plotting applications.

Provides common utilities for range calculation, radar parameter extraction,
and UI widget creation.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def extract_radar_params(radar_cfg: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Extract and validate radar configuration parameters.
    
    Args:
        radar_cfg: Radar configuration dictionary
        
    Returns:
        Tuple of (sample_rate_hz, start_freq_hz, stop_freq_hz, chirp_duration_s)
        Returns None for any parameter that is missing or invalid
    """
    sample_rate_hz = radar_cfg.get("adc_sample_rate_Hz")
    if isinstance(sample_rate_hz, (int, float)) and sample_rate_hz > 0:
        sample_rate_hz = float(sample_rate_hz)
    else:
        sample_rate_hz = None
    
    start_freq_hz = radar_cfg.get("start_frequency_Hz")
    if isinstance(start_freq_hz, (int, float)) and start_freq_hz > 0:
        start_freq_hz = float(start_freq_hz)
    else:
        start_freq_hz = None
    
    stop_freq_hz = radar_cfg.get("stop_frequency_Hz")
    if isinstance(stop_freq_hz, (int, float)) and stop_freq_hz > 0:
        stop_freq_hz = float(stop_freq_hz)
    else:
        stop_freq_hz = None
    
    chirp_duration_s = radar_cfg.get("chirp_duration_s")
    if isinstance(chirp_duration_s, (int, float)) and chirp_duration_s > 0:
        chirp_duration_s = float(chirp_duration_s)
    else:
        chirp_duration_s = None
    
    return sample_rate_hz, start_freq_hz, stop_freq_hz, chirp_duration_s


def calculate_range_axis(
    num_bins: int,
    sample_rate_hz: float,
    start_freq_hz: float,
    stop_freq_hz: float,
    chirp_duration_s: float,
    use_k_chirp: bool = False,
    k_chirp: Optional[float] = None
) -> np.ndarray:
    """
    Calculate range axis in cm from FMCW parameters.
    
    Args:
        num_bins: Number of frequency bins (positive half only)
        sample_rate_hz: ADC sample rate in Hz
        start_freq_hz: Start frequency in Hz
        stop_freq_hz: Stop frequency in Hz
        chirp_duration_s: Chirp duration in seconds
        use_k_chirp: If True, use k_chirp directly instead of calculating from bandwidth
        k_chirp: Chirp slope in Hz/s (required if use_k_chirp is True)
        
    Returns:
        Range axis array in cm
    """
    # Calculate frequency resolution
    N = num_bins * 2  # Full FFT size (positive + negative frequencies)
    freq_resolution = sample_rate_hz / N
    freq_axis_positive = np.arange(num_bins) * freq_resolution
    
    # Calculate chirp slope
    if use_k_chirp and k_chirp is not None:
        slope_hz_per_s = k_chirp
    else:
        bandwidth_hz = stop_freq_hz - start_freq_hz
        slope_hz_per_s = bandwidth_hz / chirp_duration_s
    
    # Convert beat frequency to range: R = (c * f_b) / (2 * slope)
    c = 3e8  # speed of light in m/s
    range_m = (c * freq_axis_positive) / (2.0 * slope_hz_per_s)
    range_axis = range_m * 100.0  # Convert to cm
    
    return range_axis


def setup_pause_button(fig, position: Tuple[float, float, float, float] = (0.85, 0.02, 0.12, 0.04), logger: Optional[Any] = None) -> Tuple[Button, Any]:
    """
    Create and configure a pause/resume button.
    
    Args:
        fig: Matplotlib figure
        position: Button position as [left, bottom, width, height]
        logger: Optional logger instance for logging pause/resume events
        
    Returns:
        Tuple of (button, get_pause_state_function)
        The get_pause_state function returns the current pause state
    """
    button_ax = plt.axes(position)
    pause_button = Button(button_ax, 'Pause', color='lightblue', hovercolor='lightcyan')
    
    is_paused = [False]  # Use list to allow modification in closure
    
    def toggle_pause(event):
        """Toggle pause/resume state"""
        is_paused[0] = not is_paused[0]
        
        if is_paused[0]:
            pause_button.label.set_text('Resume')
            pause_button.color = 'lightcoral'
            pause_button.hovercolor = 'lightpink'
            if logger:
                logger.info("Data updates paused")
        else:
            pause_button.label.set_text('Pause')
            pause_button.color = 'lightblue'
            pause_button.hovercolor = 'lightcyan'
            if logger:
                logger.info("Data updates resumed")
        
        # Redraw button
        pause_button.ax.figure.canvas.draw_idle()
    
    pause_button.on_clicked(toggle_pause)
    
    # Return button and a getter function for pause state
    def get_pause_state():
        return is_paused[0]
    
    return pause_button, get_pause_state

