"""
Live range-profile plot of FMCW radar frames from ZeroMQ.

Subscribes to data and status PUB channels and sends commands on a REQ
command channel. Plots log-magnitude range profile after:
1. Windowing the time-domain signals
2. Computing FFT
3. Non-Coherent Integration (NCI) across all three channels
4. Converting to log-magnitude
5. Converting beat frequency bins to physical range in cm

Displays physical units:
- X-axis: Range in cm (calculated from FMCW chirp parameters)
- Y-axis: Magnitude in dB

Optimized for >10 Hz updates by draining the ZMQ socket and only plotting
the latest frame.

Usage:
  python plot_range.py --config range.yaml

"""

import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import logging

from piradar.client.config import FrequencyDomainConfig, load_config
from piradar.client.plotting import BaseRadarPlotter, calculate_range_axis
from piradar.client.communication import drain_latest_frame_full, drain_latest_frame
from piradar.client.processing.peak_finder import find_highest_peak
from piradar.client.processing import apply_window_1d, compute_fft_spectrum


class RangePlotter(BaseRadarPlotter):
    """Range profile plotter with H5 streaming support."""
    
    def __init__(self, config: FrequencyDomainConfig, log_level: str = "INFO"):
        super().__init__(config, log_level=log_level)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.k_chirp: Optional[float] = None
        
        # H5 streaming state
        self.is_streaming = False
        self.h5_file: Optional[h5py.File] = None
        self.h5_filename: Optional[str] = None
        self.h5_frame_ids = []
        self.h5_timestamps = []
        self.h5_data_chunks = []
        self.stream_start_time: Optional[float] = None
        
        # Plot elements
        self.line = None
        self.peak_marker = None
        self.rx0_line = None
        self.rx1_line = None
        self.rx2_line = None
        self.ax = None
        self.stream_button = None
    
    def connect(self) -> bool:
        """Connect and extract k_chirp parameter."""
        if not super().connect():
            return False
        
        # Extract k_chirp if available
        if self.radar_cfg:
            k_chirp = self.radar_cfg.get("k_chirp")
            if isinstance(k_chirp, (int, float)) and k_chirp > 0:
                self.k_chirp = float(k_chirp)
        
        # Validate required parameters
        if not all([self.sample_rate_hz, self.start_freq_hz, self.stop_freq_hz, self.chirp_duration_s]):
            self.logger.error("Invalid or missing FMCW parameters in radar config.")
            return False
        
        if not self.k_chirp:
            self.logger.error("Invalid or missing k_chirp in radar config.")
            return False
        
        self.logger.info(f"Window: {self.config.plot.window}, Dynamic range: {self.config.plot.db_min} dB - {self.config.plot.db_max} dB")
        
        if self.config.peak_finding.enabled:
            self.logger.info(f"Peak finding enabled - Min range: {self.config.peak_finding.min_range_cm} cm, Threshold: {self.config.peak_finding.threshold}")
        else:
            self.logger.info("Peak finding disabled")
        
        return True
    
    def setup_plot(self) -> None:
        """Setup matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title(f"Radar Range Profile (Window: {self.config.plot.window})")
        self.ax.set_xlabel("Range (cm)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.grid(True, alpha=0.3)
        
        # Initialize empty lines and peak marker
        label = "Coherent Integration" if self.config.integration_method == "coherent" else "NCI"
        self.line, = self.ax.plot([], [], 'b-', lw=1.5, label=label)
        self.peak_marker, = self.ax.plot([], [], 'rx', markersize=10, markeredgewidth=2, label="Peak")
        
        # Individual channel lines for debugging
        self.rx0_line, = self.ax.plot([], [], 'r-', lw=1, alpha=0.7, label="RX0")
        self.rx1_line, = self.ax.plot([], [], 'g-', lw=1, alpha=0.7, label="RX1")
        self.rx2_line, = self.ax.plot([], [], 'm-', lw=1, alpha=0.7, label="RX2")
        
        self.ax.legend(loc="upper right")
        
        # Add streaming button
        stream_button_ax = plt.axes([0.70, 0.02, 0.12, 0.04])
        self.stream_button = Button(stream_button_ax, 'Start Stream', color='lightgreen', hovercolor='lightyellow')
        self.stream_button.on_clicked(self.toggle_streaming)
    
    def toggle_streaming(self, event) -> None:
        """Toggle streaming to h5 file."""
        if not self.is_streaming:
            # Start streaming
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.h5_filename = f"radar_data_{timestamp_str}.h5"
            
            try:
                self.h5_file = h5py.File(self.h5_filename, 'w')
                
                # Write metadata (radar configuration)
                if self.radar_cfg:
                    metadata_group = self.h5_file.create_group('metadata')
                    for key, value in self.radar_cfg.items():
                        if isinstance(value, (int, float, str, bool)):
                            metadata_group.attrs[key] = value
                        elif isinstance(value, (list, tuple)):
                            metadata_group.attrs[key] = np.array(value)
                    metadata_group.attrs['recording_start_time'] = datetime.now().isoformat()
                    metadata_group.attrs['sample_rate_hz'] = self.sample_rate_hz
                    metadata_group.attrs['start_frequency_Hz'] = self.start_freq_hz
                    metadata_group.attrs['stop_frequency_Hz'] = self.stop_freq_hz
                    metadata_group.attrs['chirp_duration_s'] = self.chirp_duration_s
                    metadata_group.attrs['chirp_index'] = self.config.chirp_index
                    metadata_group.attrs['window_type'] = self.config.plot.window
                    metadata_group.attrs['integration_method'] = self.config.integration_method
                
                # Initialize datasets
                self.h5_file.create_dataset('frame_ids', (0,), maxshape=(None,), dtype=np.int64)
                self.h5_file.create_dataset('timestamps', (0,), maxshape=(None,), dtype=np.float64)
                
                self.h5_frame_ids = []
                self.h5_timestamps = []
                self.h5_data_chunks = []
                self.stream_start_time = time.time()
                
                self.is_streaming = True
                self.stream_button.label.set_text('Stop Stream')
                self.stream_button.color = 'lightcoral'
                self.stream_button.hovercolor = 'lightpink'
                self.logger.info(f"Started streaming to {self.h5_filename}")
                
            except Exception as e:
                self.logger.error(f"Failed to start streaming: {e}")
                if self.h5_file is not None:
                    try:
                        self.h5_file.close()
                    except Exception:
                        pass
                    self.h5_file = None
                return
        else:
            # Stop streaming
            if self.h5_file is not None:
                try:
                    if self.h5_frame_ids:
                        num_frames = len(self.h5_frame_ids)
                        
                        # Resize datasets
                        self.h5_file['frame_ids'].resize((num_frames,))
                        self.h5_file['timestamps'].resize((num_frames,))
                        
                        # Write frame IDs and timestamps
                        self.h5_file['frame_ids'][:] = np.array(self.h5_frame_ids, dtype=np.int64)
                        self.h5_file['timestamps'][:] = np.array(self.h5_timestamps, dtype=np.float64)
                        
                        # Write data chunks
                        if self.h5_data_chunks:
                            first_chunk_shape = self.h5_data_chunks[0].shape
                            data_shape = (num_frames,) + first_chunk_shape
                            self.h5_file.create_dataset('data', data_shape, dtype=self.h5_data_chunks[0].dtype)
                            for i, chunk in enumerate(self.h5_data_chunks):
                                self.h5_file['data'][i] = chunk
                        
                        # Add recording metadata
                        if 'metadata' in self.h5_file:
                            duration = time.time() - self.stream_start_time if self.stream_start_time else 0.0
                            self.h5_file['metadata'].attrs['recording_duration_seconds'] = duration
                            self.h5_file['metadata'].attrs['num_frames'] = num_frames
                            self.h5_file['metadata'].attrs['recording_end_time'] = datetime.now().isoformat()
                            if num_frames > 0:
                                self.h5_file['metadata'].attrs['average_fps'] = num_frames / duration if duration > 0 else 0.0
                    
                    saved_filename = self.h5_filename
                    self.h5_file.close()
                    self.logger.info(f"Stopped streaming. Saved {len(self.h5_frame_ids)} frames to {saved_filename}")
                    
                except Exception as e:
                    self.logger.error(f"Error closing h5 file: {e}")
                finally:
                    self.h5_file = None
                    self.h5_filename = None
                    self.h5_frame_ids = []
                    self.h5_timestamps = []
                    self.h5_data_chunks = []
            
            self.is_streaming = False
            self.stream_button.label.set_text('Start Stream')
            self.stream_button.color = 'lightgreen'
            self.stream_button.hovercolor = 'lightyellow'
        
        # Redraw button
        self.stream_button.ax.figure.canvas.draw_idle()
    
    def drain_frame(self) -> Optional[Any]:
        """Drain frame - use full frame dict if streaming for H5 metadata."""
        if self.is_streaming:
            return drain_latest_frame_full(
                self.data_sock, self.data_poller, max_drain=self.config.zmq.max_drain
            )
        else:
            return drain_latest_frame(
                self.data_sock, self.data_poller, max_drain=self.config.zmq.max_drain
            )
    
    def process_frame(self, frame_data: np.ndarray, frame_metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Process a single radar frame."""
        
        # Extract time series
        if self.config.chirp_index < 0:
            ts = np.mean(frame_data, axis=0)
        else:
            ts = frame_data[self.config.chirp_index]
        
        # Process frequency domain
        windowed = apply_window_1d(ts, self.config.plot.window)
        complex_spectrum = compute_fft_spectrum(windowed)
        
        N = complex_spectrum.shape[0]
        positive_half_idx = N // 2
        complex_spectrum_positive = complex_spectrum[positive_half_idx:]
        
        # Integration
        if self.config.integration_method == "coherent":
            integrated_spectrum = np.power(np.abs(np.sum(complex_spectrum_positive, axis=1)), 2)
        else:
            integrated_spectrum = np.sum(np.power(np.abs(complex_spectrum_positive), 2), axis=1)
        
        log_mag_spectrum = 10 * np.log10(integrated_spectrum + 1e-12)
        
        # Calculate range axis
        range_axis = calculate_range_axis(
            len(log_mag_spectrum),
            self.sample_rate_hz,
            self.start_freq_hz,
            self.stop_freq_hz,
            self.chirp_duration_s,
            use_k_chirp=True,
            k_chirp=self.k_chirp
        )
        
        # Peak finding
        peak_result = None
        if self.config.peak_finding.enabled:
            method = 'parabolic' if self.config.integration_method == "coherent" else 'centroid'
            complex_spectrum_sum = np.sqrt(integrated_spectrum)
            peak_result = find_highest_peak(
                complex_spectrum_sum,
                range_axis,
                ignore_range_0=self.config.peak_finding.ignore_range_0,
                min_range_cm=self.config.peak_finding.min_range_cm,
                threshold=self.config.peak_finding.threshold,
                method=method
            )
        else:
            peak_result = {'peak_position': None, 'peak_range_cm': None, 'peak_amplitude': None}
        
        # Store frame for H5 streaming
        if self.is_streaming and self.h5_file is not None and frame_metadata is not None:
            try:
                self.h5_frame_ids.append(frame_metadata.get('frame_id', -1))
                self.h5_timestamps.append(frame_metadata.get('timestamp', time.time()))
                # Store original frame data (before processing)
                original_frame = frame_metadata.get('data', frame_data)
                self.h5_data_chunks.append(original_frame.copy())
            except Exception as e:
                self.logger.error(f"Error storing frame in h5 buffer: {e}")
        
        return {
            'range_axis': range_axis,
            'log_mag_spectrum': log_mag_spectrum,
            'complex_spectrum_positive': complex_spectrum_positive,
            'peak_result': peak_result,
            'bandwidth_hz': self.stop_freq_hz - self.start_freq_hz,
            'slope_hz_per_s': self.k_chirp
        }
    
    def update_plot(self, processed_data: Dict[str, Any]) -> None:
        """Update the plot with processed data."""
        range_axis = processed_data['range_axis']
        log_mag_spectrum = processed_data['log_mag_spectrum']
        complex_spectrum_positive = processed_data['complex_spectrum_positive']
        peak_result = processed_data['peak_result']
        bandwidth_hz = processed_data['bandwidth_hz']
        slope_hz_per_s = processed_data['slope_hz_per_s']
        
        # Update main plot
        self.line.set_data(range_axis, log_mag_spectrum)
        
        # Update individual channel plots
        if self.config.show_individual_channels:
            log_mag_channels = 20 * np.log10(np.abs(complex_spectrum_positive) + 1e-12)
            self.rx0_line.set_data(range_axis, log_mag_channels[:, 0])
            self.rx1_line.set_data(range_axis, log_mag_channels[:, 1])
            self.rx2_line.set_data(range_axis, log_mag_channels[:, 2])
            self.rx0_line.set_visible(True)
            self.rx1_line.set_visible(True)
            self.rx2_line.set_visible(True)
        else:
            self.rx0_line.set_visible(False)
            self.rx1_line.set_visible(False)
            self.rx2_line.set_visible(False)
        
        # Update peak marker
        if peak_result['peak_position'] is not None and peak_result['peak_range_cm'] is not None:
            peak_amplitude_log = 20 * np.log10(np.abs(peak_result['peak_amplitude']))
            self.peak_marker.set_data([peak_result['peak_range_cm']], [peak_amplitude_log])
            self.peak_marker.set_visible(True)
            self.ax.set_title(
                f"Radar Range Profile (Window: {self.config.plot.window}) - "
                f"Peak: {peak_result['peak_range_cm']:.1f} cm, {peak_amplitude_log:.1f} dB"
            )
        else:
            self.peak_marker.set_visible(False)
            self.ax.set_title(f"Radar Range Profile (Window: {self.config.plot.window}) - No Peak")
        
        # Set axis limits
        self.ax.set_xlim(range_axis[0], range_axis[-1])
        self.ax.set_ylim(self.config.plot.db_min, self.config.plot.db_max)
        
        # Update x-axis label
        x_label = f"Range (cm) [BW: {bandwidth_hz/1e9:.2f} GHz, Slope: {slope_hz_per_s/1e12:.1f} MHz/usec]"
        self.ax.set_xlabel(x_label)
        
        # Efficient redraw
        self.fig.canvas.draw_idle()
    
    def cleanup(self) -> None:
        """Cleanup including H5 streaming."""
        # Stop streaming if active
        if self.is_streaming and self.h5_file is not None:
            self.toggle_streaming(None)
        
        # Call parent cleanup
        super().cleanup()


def main() -> int:
    """Main entry point."""
    args = RangePlotter.parse_args("Live plot of range profile radar data from ZMQ", "range.yaml")
    
    # Load configuration
    try:
        config = load_config(args.config, FrequencyDomainConfig)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return 1
    
    # Create and run plotter
    plotter = RangePlotter(config, log_level=args.log_level)
    return plotter.run()


if __name__ == "__main__":
    sys.exit(main())
