"""
Antenna calibration script.

This script connects to the radar and the stepper motor controller and measures the peak signal strength at a given distance.
"""

import time
import argparse
import sys
import yaml
import math
import logging
import numpy as np
import zmq
from cmdstepper import StepperController
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend for better performance
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
from pydantic import BaseModel

from piradar.client.config import FrequencyDomainConfig, load_config
from piradar.client.communication import (
    connect_sub, drain_latest_frame, drain_latest_status,
    send_command, start_radar_streaming, stop_radar_streaming, HttpClient,
    get_radar_config
)
from piradar.client.plotting.helpers import extract_radar_params, calculate_range_axis
from piradar.client.processing import (
    extract_time_series, process_frequency_domain, calculate_frequency_axis,
    apply_window_1d, compute_fft_spectrum, to_log_magnitude
)
from piradar.client.processing.peak_finder import find_highest_peak
from piradar.client.logging_utils import configure_logging




class StepperConfig(BaseModel):
    hostname: str
    home_position: int = 86
    steps: int = 400
    speed: int
    port: str = "/dev/ttyUSB0"
    baudrate: int = 115200
    timeout: float = 1.0


class CalibrationConfig(BaseModel):
    stepper: StepperConfig
    start: float
    stop: float
    step: float
    pause: float
    distance: float
    calibrate: bool = False
    
    # Radar configuration
    zmq_host: str = "localhost"
    zmq_data_port: int = 5555
    zmq_status_port: int = 5557  # ZMQ status publisher port
    zmq_command_port: int = 5556  # HTTP command port
    zmq_max_drain: int = 20
    command_timeout: float = 2.0  # HTTP command timeout in seconds
    average: bool = True
    window: str = "hann"
    db_range: int = 80
    test_mode: bool = False  # If True, skip radar measurements
    averaging: int = 20
    
    # Plotting configuration
    enable_plotting: bool = True  # Enable live plotting during measurements
    show_individual_channels: bool = False  # Show individual RX channel plots
    peak_finding_enabled: bool = True  # Enable peak finding
    peak_finding_min_range_cm: float = 5.0  # Minimum range for peak finding
    peak_finding_threshold: Optional[float] = None  # Peak detection threshold
    peak_finding_ignore_range_0: bool = True  # Ignore range 0 (self-coupling)
    peak_finding_distance_tolerance_cm: float = 10.0  # Tolerance around target distance for peak finding

logger = logging.getLogger(__name__)


class RadarMeasurement:
    def __init__(self, config: CalibrationConfig):
        """Initialize radar measurement system."""
        self.config = config
        self.ctx = None
        self.data_sock = None
        self.data_poller = None
        self.status_sock = None
        self.status_poller = None
        self.cmd_sock = None
        self.radar_cfg = None
        self.sample_rate_hz = None
        self.start_freq_hz = None
        self.stop_freq_hz = None
        self.chirp_duration_s = None
        self.k_chirp = None
        
        # Plotting setup
        self.fig = None
        self.ax = None
        self.line = None
        self.peak_marker = None
        self.rx0_line = None
        self.rx1_line = None
        self.rx2_line = None
        self.range_axis = None
        
    def connect(self):
        """Connect to radar system via ZMQ."""
        logger.info(f"Connecting to radar at {self.config.zmq_host}...")
        logger.debug(f"ZMQ ports: data={self.config.zmq_data_port}, status={self.config.zmq_status_port}, command={self.config.zmq_command_port}")
        try:
            self.ctx = zmq.Context()
            logger.debug("Creating ZMQ data subscriber socket...")
            self.data_sock, self.data_poller = connect_sub(self.ctx, self.config.zmq_host, self.config.zmq_data_port)
            logger.debug("Creating ZMQ status subscriber socket...")
            self.status_sock, self.status_poller = connect_sub(self.ctx, self.config.zmq_host, self.config.zmq_status_port)
            # Use HttpClient with default timeout (matches BaseRadarPlotter and other working examples)
            logger.debug("Creating HTTP command client...")
            self.cmd_sock = HttpClient(self.config.zmq_host, self.config.zmq_command_port)
            
            # Update config with resolved host for logging (matches BaseRadarPlotter)
            resolved_host = self.cmd_sock.host
            if resolved_host != self.config.zmq_host:
                logger.debug(f"Resolved host {self.config.zmq_host} to {resolved_host}")
            self.config.zmq_host = resolved_host

            # Get radar configuration
            logger.debug("Retrieving radar configuration from server...")
            self.radar_cfg = get_radar_config(self.cmd_sock)
            if not self.radar_cfg:
                logger.warning("Failed to retrieve radar configuration from server.")
                return False
            logger.debug("Radar configuration retrieved successfully")
            
            # Extract radar parameters using helper function
            logger.debug("Extracting radar parameters...")
            self.sample_rate_hz, self.start_freq_hz, self.stop_freq_hz, self.chirp_duration_s = extract_radar_params(self.radar_cfg)
            
            # Extract k_chirp if available
            if self.radar_cfg:
                k_chirp = self.radar_cfg.get("k_chirp")
                if isinstance(k_chirp, (int, float)) and k_chirp > 0:
                    self.k_chirp = float(k_chirp)
                    logger.debug(f"k_chirp: {self.k_chirp/1e12:.2f} MHz/usec")
                else:
                    logger.debug("k_chirp not found in radar config, will calculate from bandwidth")
            
            # Validate required parameters
            if not all([self.sample_rate_hz, self.start_freq_hz, self.stop_freq_hz, self.chirp_duration_s]):
                logger.error("Invalid or missing FMCW parameters in radar config.")
                logger.error(f"  sample_rate_hz: {self.sample_rate_hz}")
                logger.error(f"  start_freq_hz: {self.start_freq_hz}")
                logger.error(f"  stop_freq_hz: {self.stop_freq_hz}")
                logger.error(f"  chirp_duration_s: {self.chirp_duration_s}")
                return False
            
            # Log extracted parameters
            logger.info("Radar parameters:")
            if self.sample_rate_hz:
                logger.info(f"  Sample rate: {self.sample_rate_hz/1e6:.1f} MHz")
            if self.start_freq_hz and self.stop_freq_hz:
                bandwidth_ghz = (self.stop_freq_hz - self.start_freq_hz) / 1e9
                logger.info(f"  Bandwidth: {bandwidth_ghz:.2f} GHz")
                logger.info(f"  Frequency range: {self.start_freq_hz/1e9:.2f} - {self.stop_freq_hz/1e9:.2f} GHz")
            if self.chirp_duration_s:
                logger.info(f"  Chirp duration: {self.chirp_duration_s*1e6:.1f} μs")
            
            logger.info(f"Successfully connected to radar at {self.config.zmq_host}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to radar: {e}", exc_info=True)
            return False
    
    def disconnect(self):
        """Disconnect from radar system."""
        if self.data_sock:
            self.data_sock.close()
        if self.status_sock:
            self.status_sock.close()
        if self.cmd_sock:
            self.cmd_sock.close()
        if self.ctx:
            self.ctx.term()
        logger.info("Disconnected from radar")
    
    def start_streaming(self):
        """Start radar data streaming."""
        logger.info("Starting radar data streaming...")
        result = start_radar_streaming(self.cmd_sock)
        if result:
            logger.info("Radar streaming started successfully")
        else:
            logger.error("Failed to start radar streaming")
        return result
    
    def stop_streaming(self):
        """Stop radar data streaming."""
        logger.info("Stopping radar data streaming...")
        result = stop_radar_streaming(self.cmd_sock)
        if result:
            logger.info("Radar streaming stopped successfully")
        else:
            logger.warning("Failed to stop radar streaming")
        return result
    
    def setup_plotting(self):
        """Setup matplotlib plotting interface."""
        if not self.config.enable_plotting:
            return
            
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_title(f"Radar Range Profile - Antenna Calibration")
        self.ax.set_xlabel("Range (cm)")
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.grid(True, alpha=0.3)

        # Initialize empty lines and peak marker
        self.line, = self.ax.plot([], [], 'b-', lw=1.5, label="Combined Range Profile")
        self.peak_marker, = self.ax.plot([], [], 'rx', markersize=10, markeredgewidth=2, label="Peak")
        
        # Individual channel lines for debugging
        self.rx0_line, = self.ax.plot([], [], 'r-', lw=1, alpha=0.7, label="RX0")
        self.rx1_line, = self.ax.plot([], [], 'g-', lw=1, alpha=0.7, label="RX1") 
        self.rx2_line, = self.ax.plot([], [], 'm-', lw=1, alpha=0.7, label="RX2")
        
        self.ax.legend(loc="upper right")
        logger.info("Plotting interface initialized")
    
    def close_plotting(self):
        """Close matplotlib plotting interface."""
        if self.fig and plt.fignum_exists(self.fig.number):
            plt.close(self.fig)
            logger.info("Plotting interface closed")
    
    def plot_range_profile(self, averaged_complex_spectra: list, averaged_log_spectrum: np.ndarray, 
                          peak_result: Optional[dict] = None, angle: float = None):
        """Plot the range profile after averaging multiple frames."""
        if not self.config.enable_plotting or self.fig is None:
            return
            
        # Calculate range axis using helper function (same as plot_range.py)
        range_axis = calculate_range_axis(
            len(averaged_log_spectrum),
            self.sample_rate_hz,
            self.start_freq_hz,
            self.stop_freq_hz,
            self.chirp_duration_s,
            use_k_chirp=(self.k_chirp is not None),
            k_chirp=self.k_chirp
        )
        self.range_axis = range_axis
        
        # Calculate slope for display
        if self.k_chirp is not None:
            slope_hz_per_s = self.k_chirp
        else:
            bandwidth_hz = self.stop_freq_hz - self.start_freq_hz
            slope_hz_per_s = bandwidth_hz / self.chirp_duration_s
        
        # Update main plot
        self.line.set_data(range_axis, averaged_log_spectrum)
        
        # Update individual channel plots (if enabled)
        if self.config.show_individual_channels and len(averaged_complex_spectra) >= 3:
            # Convert complex spectra to log magnitude for individual channels
            # Use 20 * log10 to match plot_range.py and be consistent with combined profile
            for i, complex_spectrum in enumerate(averaged_complex_spectra):
                log_mag_channel = 20 * np.log10(np.abs(complex_spectrum) + 1e-12)
                if i == 0:
                    self.rx0_line.set_data(range_axis, log_mag_channel)
                    self.rx0_line.set_visible(True)
                elif i == 1:
                    self.rx1_line.set_data(range_axis, log_mag_channel)
                    self.rx1_line.set_visible(True)
                elif i == 2:
                    self.rx2_line.set_data(range_axis, log_mag_channel)
                    self.rx2_line.set_visible(True)
        else:
            # Hide individual channel lines
            self.rx0_line.set_visible(False)
            self.rx1_line.set_visible(False)
            self.rx2_line.set_visible(False)
        
        # Update peak marker
        if peak_result and peak_result.get('peak_range_cm') is not None:
            # Convert peak amplitude from magnitude to log magnitude for display (matching plot_range.py)
            peak_amplitude_log = 20 * np.log10(np.abs(peak_result['peak_amplitude']))
            self.peak_marker.set_data([peak_result['peak_range_cm']], [peak_amplitude_log])
            self.peak_marker.set_visible(True)
            title_suffix = f" - Peak: {peak_result['peak_range_cm']:.1f} cm, {peak_amplitude_log:.1f} dB"
        else:
            self.peak_marker.set_visible(False)
            title_suffix = " - No Peak"
        
        # Update title with angle information
        angle_info = f" at {angle:.1f}°" if angle is not None else ""
        self.ax.set_title(f"Radar Range Profile - Antenna Calibration{angle_info}{title_suffix}")
        
        # Set axis limits
        self.ax.set_xlim(range_axis[0], range_axis[-1])
        
        # Calculate Y-axis limits based on db_range and data
        # Find the maximum value in the combined spectrum
        max_db = np.max(averaged_log_spectrum)
        # Set Y-axis range: max_db down to (max_db - db_range)
        db_min = max_db - self.config.db_range
        db_max = max_db + 5  # Add small margin at top
        self.ax.set_ylim(db_min, db_max)
        logger.debug(f"Y-axis limits: {db_min:.1f} to {db_max:.1f} dB (range: {self.config.db_range} dB, max: {max_db:.1f} dB)")
        
        # Update x-axis label with FMCW parameters
        bandwidth_hz = self.stop_freq_hz - self.start_freq_hz
        x_label = f"Range (cm) [BW: {bandwidth_hz/1e9:.2f} GHz, Slope: {slope_hz_per_s/1e12:.1f} MHz/usec]"
        self.ax.set_xlabel(x_label)

        # Efficient redraw
        self.fig.canvas.draw_idle()
        plt.pause(0.01)
    
    def measure_peak(self) -> Optional[dict]:
        """Measure complex peak values from all three receivers around the target distance."""
        logger.debug(f"Starting peak measurement with {self.config.averaging} frame averaging")
        # Collect complex data from all three receivers over multiple frames
        all_complex_spectra = []

        for i in range(self.config.averaging):
            latest = drain_latest_frame(self.data_sock, self.data_poller, max_drain=1)

            if latest is None:
                logger.error(f"Frame {i+1}/{self.config.averaging} is empty. Returning None.")
                return None
            
            logger.debug(f"Processing frame {i+1}/{self.config.averaging}")

            # compute mean over slowtime
            ts = latest.mean(axis=0)

            # Process frequency domain
            windowed = apply_window_1d(ts, self.config.window)
            complex_spectrum = compute_fft_spectrum(windowed)
            
            N = complex_spectrum.shape[0]
            positive_half_idx = N // 2
            complex_spectrum_positive = complex_spectrum[positive_half_idx:]
            all_complex_spectra.append(complex_spectrum_positive)

       
        # Average complex spectra over all frames, shape (fasttime, rx)
        averaged_complex_spectra = np.mean(all_complex_spectra, axis=0)
        logger.debug(f"Averaged complex spectra shape: {averaged_complex_spectra.shape}")

        # NCI - sum over rx channels (axis=1), result shape (fasttime,)
        integrated_spectrum = np.sum(np.power(np.abs(averaged_complex_spectra), 2), axis=1)
        averaged_log_spectrum = 10 * np.log10(integrated_spectrum + 1e-12)

        logger.debug(f"Averaged log spectrum shape: {averaged_log_spectrum.shape}, min: {np.min(averaged_log_spectrum):.2f} dB, max: {np.max(averaged_log_spectrum):.2f} dB")
        
        # Calculate range axis using helper function (same as plot_range.py)
        range_axis = calculate_range_axis(
            len(averaged_log_spectrum),
            self.sample_rate_hz,
            self.start_freq_hz,
            self.stop_freq_hz,
            self.chirp_duration_s,
            use_k_chirp=(self.k_chirp is not None),
            k_chirp=self.k_chirp
        )
        
        # Calculate slope for display
        if self.k_chirp is not None:
            slope_hz_per_s = self.k_chirp
        else:
            bandwidth_hz = self.stop_freq_hz - self.start_freq_hz
            slope_hz_per_s = bandwidth_hz / self.chirp_duration_s


        # Peak finding in desired range (around target distance with tolerance)
        peak_result = None
        method = 'centroid'
        complex_spectrum_sum = np.sqrt(integrated_spectrum)
        
        # Calculate min and max range based on target distance ± tolerance
        target_distance_cm = self.config.distance
        tolerance_cm = self.config.peak_finding_distance_tolerance_cm
        min_range_cm = max(self.config.peak_finding_min_range_cm, target_distance_cm - tolerance_cm)
        max_range_cm = target_distance_cm + tolerance_cm
        
        logger.debug(f"Peak finding: searching in range {min_range_cm:.1f} - {max_range_cm:.1f} cm (target: {target_distance_cm:.1f} cm ± {tolerance_cm:.1f} cm)")
        
        peak_result = find_highest_peak(
            complex_spectrum_sum,
            range_axis,
            ignore_range_0=self.config.peak_finding_ignore_range_0,
            min_range_cm=min_range_cm,
            max_range_cm=max_range_cm,
            threshold=self.config.peak_finding_threshold,
            method=method
        )
        
        if peak_result and peak_result.get('peak_range_cm') is not None:
            logger.debug(f"Peak found at {peak_result['peak_range_cm']:.1f} cm")
        else:
            logger.debug("Peak finding did not find a valid peak")
        
        # Fallback to original method if peak finding is disabled or fails
        if not peak_result or peak_result.get('peak_range_cm') is None:
            logger.warning("No peak found using peak finding algorithm. Using fallback method around target distance.")
            # Find peak around target distance using log spectrum
            target_distance_cm = self.config.distance
            tolerance_cm = 10.0
            logger.debug(f"Searching for peak around target distance {target_distance_cm} cm ± {tolerance_cm} cm")
            
            distance_mask = (range_axis >= (target_distance_cm - tolerance_cm)) & (range_axis <= (target_distance_cm + tolerance_cm))
            
            if not np.any(distance_mask):
                logger.error(f"No range bins found within {tolerance_cm} cm of target distance {target_distance_cm} cm")
                return None
            
            log_spectrum_in_range = averaged_log_spectrum[distance_mask]
            if len(log_spectrum_in_range) == 0:
                logger.error("Empty log spectrum in target range")
                return None
            
            # Find peak bin
            peak_bin_in_range = np.argmax(log_spectrum_in_range)
            range_indices = np.where(distance_mask)[0]
            peak_bin = range_indices[peak_bin_in_range]
            actual_distance = range_axis[peak_bin]
            peak_log_value = log_spectrum_in_range[peak_bin_in_range]
            
            logger.info(f"Fallback method found peak at {actual_distance:.1f} cm with {peak_log_value:.2f} dB")
            
            # Create a simple peak result structure
            peak_result = {
                'peak_range_cm': actual_distance,
                'peak_amplitude': peak_log_value,
                'peak_position': peak_bin
            }
        else:
            actual_distance = peak_result['peak_range_cm']
            # Convert peak amplitude from complex magnitude to log magnitude
            peak_log_value = 20 * np.log10(np.abs(peak_result['peak_amplitude']))
            logger.debug(f"Peak finding successful: {actual_distance:.1f} cm, {peak_log_value:.2f} dB")
        
        # Extract complex values at peak bin for all three receivers
        # averaged_complex_spectra has shape (fasttime, rx), so index as [peak_bin, receiver_idx]
        peak_bin = int(round(peak_result['peak_position'])) if 'peak_position' in peak_result else None
        if peak_bin is not None:
            # Ensure peak_bin is within bounds
            if peak_bin >= averaged_complex_spectra.shape[0]:
                logger.warning(f"Peak bin {peak_bin} is out of bounds (max: {averaged_complex_spectra.shape[0]-1}), clamping")
                peak_bin = averaged_complex_spectra.shape[0] - 1
            peak_complex_values = []
            num_rx = averaged_complex_spectra.shape[1]
            for receiver_idx in range(min(3, num_rx)):
                complex_value = averaged_complex_spectra[peak_bin, receiver_idx]
                peak_complex_values.append(complex_value)
            logger.debug(f"Extracted complex values at peak bin {peak_bin} for {len(peak_complex_values)} receivers")
        else:
            peak_complex_values = []
            logger.warning("No valid peak bin found, cannot extract complex values")
        
        logger.info(f"Peak measurement: {peak_log_value:.2f} dB at {actual_distance:.1f} cm")
        if peak_complex_values:
            logger.debug(f"Complex values at peak: {[f'{c:.3f}' for c in peak_complex_values]}")
            logger.debug(f"Complex magnitudes: {[f'{abs(c):.3f}' for c in peak_complex_values]}")
            logger.debug(f"Complex phases (deg): {[f'{np.angle(c)*180/np.pi:.1f}' for c in peak_complex_values]}")

        # Convert averaged_complex_spectra from (fasttime, rx) array to list of (fasttime,) arrays for plotting
        # Each element in the list corresponds to one receiver channel
        averaged_complex_spectra_list = [averaged_complex_spectra[:, i] for i in range(averaged_complex_spectra.shape[1])]
        
        return {
            'log_magnitude': float(peak_log_value),
            'distance': float(actual_distance),
            'complex_values': [complex(c.real, c.imag) for c in peak_complex_values] if peak_complex_values else [],
            'peak_result': peak_result,
            'averaged_complex_spectra': averaged_complex_spectra_list,
            'averaged_log_spectrum': averaged_log_spectrum
        }

        
def run_measurement(controller: StepperController, radar: RadarMeasurement, config: CalibrationConfig):
    """Run a measurement script based on config file."""
    logger.info("=" * 60)
    logger.info("Starting antenna calibration measurement")
    logger.info(f"Configuration: start={config.start}°, stop={config.stop}°, step={config.step}°")
    logger.info(f"Target distance: {config.distance} cm, Averaging: {config.averaging} frames")
    logger.info(f"Window: {config.window}, Peak finding: {config.peak_finding_enabled}")
    logger.info("=" * 60)
    
    # Setup motor
    logger.info("Configuring stepper motor controller...")
    controller.steps = config.stepper.steps
    controller.enable = True
    controller.speed = config.stepper.speed
    logger.debug(f"Motor steps: {controller.steps}, speed: {controller.speed}")
    
    if config.calibrate:
        logger.info("Calibrating stepper motor...")
        controller.calibrate()
        controller.home(config.stepper.home_position)
        logger.info("Motor calibration complete")
    else:
        logger.info(f"Moving motor to start position: {config.start}°")
        controller.angle = config.start

    # Start radar
    if not config.test_mode:
        logger.info("Initializing radar system...")
        radar.start_streaming()
        # Setup plotting interface
        if config.enable_plotting:
            radar.setup_plotting()
    else:
        logger.info("Test mode: Skipping radar initialization")

    # Calculate angles
    logger.debug("Calculating measurement angles...")
    angles = []
    current_angle = config.start
    while (config.step > 0 and current_angle <= config.stop) or (config.step < 0 and current_angle >= config.stop):
        angles.append(current_angle)
        current_angle += config.step
    
    if angles and ((config.step > 0 and angles[-1] < config.stop) or (config.step < 0 and angles[-1] > config.stop)):
        angles.append(config.stop)
    
    logger.info(f"Will measure at {len(angles)} angles: {angles[0]:.1f}° to {angles[-1]:.1f}°")
    
    # Measure at each angle
    measurement_data = []
    for i, angle in enumerate(angles):
        logger.info(f"[{i+1}/{len(angles)}] Moving to angle: {angle:.2f}°")
        controller.move_to_angle(angle)
        actual_angle = controller.current_angle
        logger.debug(f"Motor reached angle: {actual_angle:.2f}° (target: {angle:.2f}°)")
        logger.debug(f"Waiting {config.pause} seconds for stabilization...")
        time.sleep(config.pause)
        
        if not config.test_mode:
            logger.debug(f"Starting radar measurement at {actual_angle:.2f}°...")
            peak_data = radar.measure_peak()
            if peak_data is not None:
                # Plot the range profile after averaging
                if config.enable_plotting:
                    radar.plot_range_profile(
                        peak_data.get('averaged_complex_spectra', []),
                        peak_data.get('averaged_log_spectrum', np.array([])),
                        peak_data.get('peak_result'),
                        actual_angle
                    )
                logger.info(f"✓ Peak at {actual_angle:.2f}°: {peak_data['log_magnitude']:.2f} dB at {peak_data['distance']:.1f} cm")
            else:
                logger.warning(f"✗ No peak found at {actual_angle:.2f}°")
        else:
            logger.debug(f"Test mode: Skipping radar measurement at {actual_angle:.2f}°")
            peak_data = None
        
        # Always save measurement data
        measurement_data.append({
            'target_angle': angle,
            'actual_angle': actual_angle,
            'peak_log_db': peak_data['log_magnitude'] if peak_data else None,
            'distance': peak_data['distance'] if peak_data else None,
            'complex_values': peak_data['complex_values'] if peak_data else None,
            'timestamp': time.time()
        })
        logger.debug(f"Measurement data saved for angle {actual_angle:.2f}°")
    
    logger.info("=" * 60)
    logger.info("All measurements completed, cleaning up...")
    
    # Cleanup
    if not config.test_mode:
        logger.info("Stopping radar streaming...")
        radar.stop_streaming()
        if config.enable_plotting:
            radar.close_plotting()
    
    # Move back to 0° at the end of measurement
    logger.info("Moving motor back to 0° position...")
    controller.move_to_angle(0.0)
    logger.info("Motor returned to 0°")
    
    controller.enable = False
    logger.debug("Motor disabled")
    
    # Save data
    logger.info("Saving measurement data to files...")
    save_measurement_data(measurement_data, config)
    logger.info("=" * 60)
    logger.info("Measurement sequence completed successfully")
    logger.info("=" * 60)


def save_measurement_data(measurement_data: list, config: CalibrationConfig):
    """Save measurement data to files."""
    import json
    from datetime import datetime
    
    logger.info(f"Saving {len(measurement_data)} measurement points...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_filename = f"antenna_calibration_{timestamp}.json"
    logger.debug(f"JSON filename: {json_filename}")
    
    # Convert complex numbers to serializable format
    serializable_data = []
    for measurement in measurement_data:
        serializable_measurement = measurement.copy()
        if 'complex_values' in serializable_measurement and serializable_measurement['complex_values'] is not None:
            serializable_measurement['complex_values'] = [
                {'real': float(c.real), 'imag': float(c.imag)} for c in serializable_measurement['complex_values']
            ]
        serializable_data.append(serializable_measurement)
    
    try:
        with open(json_filename, 'w') as f:
            json.dump({
                'config': config.model_dump(),
                'measurements': serializable_data,
                'timestamp': timestamp
            }, f, indent=2)
        logger.debug(f"JSON file saved: {json_filename}")
    except Exception as e:
        logger.error(f"Failed to save JSON file: {e}")
        raise
    
    # Save as CSV
    csv_filename = f"antenna_calibration_{timestamp}.csv"
    logger.debug(f"CSV filename: {csv_filename}")
    try:
        with open(csv_filename, 'w') as f:
            f.write("target_angle,actual_angle,peak_log_db,distance,complex_rx0,complex_rx1,complex_rx2,timestamp\n")
            for measurement in measurement_data:
                peak_log = measurement.get('peak_log_db', measurement.get('peak_value_db')) if measurement.get('peak_log_db', measurement.get('peak_value_db')) is not None else ""
                distance = measurement.get('distance', '') if measurement.get('distance') is not None else ""
                complex_vals = measurement.get('complex_values', [])
                if complex_vals:
                    complex_str = f"{complex_vals[0]:.6f},{complex_vals[1]:.6f},{complex_vals[2]:.6f}"
                else:
                    complex_str = ",,"
                f.write(f"{measurement['target_angle']:.2f},{measurement['actual_angle']:.2f},{peak_log},{distance},{complex_str},{measurement['timestamp']:.3f}\n")
        logger.debug(f"CSV file saved: {csv_filename}")
    except Exception as e:
        logger.error(f"Failed to save CSV file: {e}")
        raise
    
    logger.info(f"Measurement data saved to {json_filename} and {csv_filename}")
    
    # Print summary
    valid_measurements = [m for m in measurement_data if m.get('peak_log_db', m.get('peak_value_db')) is not None]
    logger.info(f"Summary: {len(valid_measurements)}/{len(measurement_data)} measurements have valid peak data")
    if valid_measurements:
        peak_values = [m.get('peak_log_db', m.get('peak_value_db')) for m in valid_measurements]
        max_peak = max(peak_values)
        min_peak = min(peak_values)
        max_angle = next(m['actual_angle'] for m in valid_measurements if m.get('peak_log_db', m.get('peak_value_db')) == max_peak)
        logger.info(f"Peak signal range: {min_peak:.2f} - {max_peak:.2f} dB")
        logger.info(f"Maximum peak signal found at {max_angle:.2f}° with {max_peak:.2f} dB")
    else:
        logger.warning("No valid radar measurements obtained - all measurements failed")


def load_config(config_path: str, config_class: type) -> CalibrationConfig:
    """Load configuration from YAML file and validate it."""
    logger.info(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        if not config_data:
            raise ValueError("Configuration file is empty")
        
        logger.debug(f"Configuration data loaded: {list(config_data.keys())}")
        config = config_class(**config_data)
        logger.info(f"Configuration loaded and validated successfully")
        logger.debug(f"Test mode: {config.test_mode}, Averaging: {config.averaging}, Window: {config.window}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def main():
    """Main function to parse arguments and run the controller."""
    parser = argparse.ArgumentParser(description="Antenna Calibration Script")
    parser.add_argument("-c", "--config", help="Path to calibration config file", required=True)
    parser.add_argument("-l", "--log-level", help="Set the log level (DEBUG, INFO, WARNING, ERROR)", default="INFO")
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(args.log_level)
    logger.info("Antenna Calibration Script Starting")
    logger.info(f"Log level: {args.log_level}")
    
    # Create controller and radar, then connect
    try:
        config = load_config(args.config, CalibrationConfig)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    logger.info("Connecting to stepper motor controller...")
    controller = StepperController(config.stepper.port, config.stepper.hostname, config.stepper.baudrate, config.stepper.timeout)
    if not controller.connect():
        logger.error(f"Failed to connect to stepper motor at {config.stepper.hostname}:{config.stepper.port}")
        sys.exit(1)
    logger.info(f"Connected to stepper motor controller at {config.stepper.hostname}:{config.stepper.port}")
    
    radar = RadarMeasurement(config)
    if not config.test_mode:
        logger.info("Connecting to radar system...")
        if not radar.connect():
            logger.error("Failed to connect to radar system. Use test_mode: true to run without radar.")
            controller.disconnect()
            sys.exit(1)
    else:
        logger.info("Test mode: Skipping radar connection")
    
    try:
        run_measurement(controller, radar, config)
    except KeyboardInterrupt:
        logger.warning("Measurement interrupted by user")
    except Exception as e:
        logger.error(f"Error during measurement: {e}", exc_info=True)
        raise
    finally:
        logger.info("Disconnecting from devices...")
        controller.disconnect()
        radar.disconnect()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main() 
