"""
Live range-Doppler plot of radar frames from ZeroMQ.

Subscribes to tcp://<host>:<port> and plots range-Doppler map with amplitude
coded as color using matplotlib's imshow function. Displays physical units:
- Range axis: Distance in centimeters (0 to maximum range)
- Velocity axis: Velocity in m/s (centered around 0)

Processing pipeline:
1. Extract time-domain data from radar frames
2. Apply windowing (optional)
3. Perform 2D FFT (range FFT + Doppler FFT)
4. Convert bins to physical units (range in meters, velocity in m/s)
5. Display as range-Doppler map with colorbar

Optimized for real-time updates by draining the ZMQ socket and only plotting
the latest frame.

Usage:
  python zmq_plot_range_doppler.py --config config/range_doppler.yaml

Requirements:
  pip install pyzmq matplotlib numpy scipy pydantic pyyaml
"""

import sys
import time
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import logging

from piradar.client.config import RangeDopplerConfig, load_config
from piradar.client.plotting import BaseRadarPlotter
from piradar.client.processing import (
    process_range_doppler, calculate_range_axis_from_config,
    calculate_velocity_axis_from_config
)


class RangeDopplerPlotter(BaseRadarPlotter):
    """Range-Doppler plotter."""
    
    def __init__(self, config: RangeDopplerConfig, log_level: str = "INFO"):
        super().__init__(config, log_level=log_level)
        logging.getLogger("httpx").setLevel(logging.WARNING)        

        self.im = None
        self.cbar = None
        self.ax = None
        self.last_scale_update = time.time()
    
    def setup_plot(self) -> None:
        """Setup matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title(f"Range-Doppler Map")
        self.ax.set_xlabel("Range (cm)")
        self.ax.set_ylabel("Velocity (m/s)")
        
        # Initialize empty image
        self.im = self.ax.imshow(
            np.zeros((10, 10)), cmap=self.config.plot.colormap,
            aspect='auto', origin='lower'
        )
        
        # Add colorbar
        self.cbar = plt.colorbar(self.im, ax=self.ax)
        self.cbar.set_label('Magnitude (dB)')
    
    def process_frame(self, frame_data: np.ndarray, frame_metadata: Optional[dict] = None) -> Optional[dict]:
        """Process a single radar frame."""
        # Process range-Doppler
        rd_map = process_range_doppler(
            frame_data, self.config.plot.window,
            self.config.plot.range_bins, self.config.plot.doppler_bins,
            self.config.plot.db_min, self.config.plot.db_max
        )
        
        # Only use positive range bins (second half after fftshift)
        num_range_bins = rd_map.shape[1]
        positive_range_bins = num_range_bins // 2
        rd_map_positive = rd_map[:, positive_range_bins:]
        
        # Calculate physical axes using radar configuration
        range_axis = None
        velocity_axis = None
        
        # Range axis (cm) using sample rate and chirp slope
        try:
            if (
                self.sample_rate_hz and
                self.start_freq_hz and
                self.stop_freq_hz and
                self.chirp_duration_s and
                self.sample_rate_hz > 0.0 and
                self.chirp_duration_s > 0.0
            ):
                slope = (self.stop_freq_hz - self.start_freq_hz) / self.chirp_duration_s
                range_axis = calculate_range_axis_from_config(num_range_bins, self.sample_rate_hz, slope)
            else:
                # Fallback - use default values if config not available
                self.logger.warning("Radar configuration not available, using default range axis")
                range_axis = np.arange(positive_range_bins) * 0.1  # 10cm bins as fallback
        except Exception as e:
            self.logger.warning(f"Failed to compute range axis from config, using fallback: {e}")
            range_axis = np.arange(positive_range_bins) * 0.1  # 10cm bins as fallback
        
        # Velocity axis
        if self.start_freq_hz and self.stop_freq_hz:
            center_freq = 0.5 * (self.start_freq_hz + self.stop_freq_hz)
            velocity_axis = calculate_velocity_axis_from_config(
                rd_map.shape[0], self.frame_length, self.frame_duration_s, center_freq
            )
        else:
            # Fallback - use default velocity axis if config not available
            self.logger.warning("Frequency config not available, using default velocity axis")
            velocity_axis = np.fft.fftshift(np.fft.fftfreq(rd_map.shape[0])) * 10.0  # Default velocity scale
        
        return {
            'rd_map_positive': rd_map_positive,
            'range_axis': range_axis,
            'velocity_axis': velocity_axis
        }
    
    def update_plot(self, processed_data: dict) -> None:
        """Update the plot with processed data."""
        rd_map_positive = processed_data['rd_map_positive']
        range_axis = processed_data['range_axis']
        velocity_axis = processed_data['velocity_axis']
        
        extent = [
            float(range_axis[0]), float(range_axis[-1]),
            float(velocity_axis[0]), float(velocity_axis[-1])
        ]
        self.im.set_data(rd_map_positive)
        self.im.set_extent(extent)
        self.im.set_clim(vmin=self.config.plot.db_min, vmax=self.config.plot.db_max)
        
        # Update axis labels with physical units
        self.ax.set_xlabel(f"Range (cm) - Range: [{range_axis[0]:.1f}, {range_axis[-1]:.1f}] cm")
        self.ax.set_ylabel(f"Velocity (m/s) - Range: [{velocity_axis[0]:.2f}, {velocity_axis[-1]:.2f}] m/s")
        
        # Efficient redraw
        now = time.time()
        if now - self.last_scale_update >= 1.0:
            self.ax.relim()
            self.ax.autoscale_view()
            self.last_scale_update = now
        
        self.fig.canvas.draw_idle()


def main() -> int:
    """Main entry point."""
    args = RangeDopplerPlotter.parse_args("Live range-Doppler plot of radar data from ZMQ", "range_doppler.yaml")
    
    # Load configuration
    try:
        config = load_config(args.config, RangeDopplerConfig)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return 1
    
    # Create and run plotter
    plotter = RangeDopplerPlotter(config, log_level=args.log_level)
    return plotter.run()


if __name__ == "__main__":
    sys.exit(main())
