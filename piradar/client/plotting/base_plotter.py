"""
Base class for radar plotting applications.

Provides common functionality for ZMQ connection, radar configuration,
streaming management, and main loop infrastructure.
"""

import argparse
import logging
import sys
import time
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod

import numpy as np
import zmq
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend for better performance
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from piradar.client.communication import (
    connect_sub, drain_latest_frame, drain_latest_status,
    start_radar_streaming, stop_radar_streaming, get_radar_config,
    HttpClient, KeepAliveDaemon
)
from piradar.client.plotting.helpers import extract_radar_params, setup_pause_button


class BaseRadarPlotter(ABC):
    """
    Base class for radar plotting applications.
    
    Handles common functionality:
    - ZMQ connection setup and teardown
    - Radar configuration retrieval
    - Streaming start/stop
    - Keep-alive daemon management
    - Main loop infrastructure
    - Pause button
    - FPS tracking
    """
    
    def __init__(self, config, log_level: str = "INFO"):
        """
        Initialize the base plotter.
        
        Args:
            config: Configuration object (must have zmq attribute)
            log_level: Logging level
        """
        self.config = config
        self.log_level = log_level
        
        # Configure logging
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f'Invalid log level: {log_level}')
        logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(message)s")
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # ZMQ connections
        self.ctx: Optional[zmq.Context] = None
        self.data_sock: Optional[zmq.Socket] = None
        self.data_poller: Optional[zmq.Poller] = None
        self.status_sock: Optional[zmq.Socket] = None
        self.status_poller: Optional[zmq.Poller] = None
        self.cmd_sock: Optional[HttpClient] = None
        self.command_available: bool = True
        
        # Radar configuration
        self.radar_cfg: Optional[Dict[str, Any]] = None
        self.sample_rate_hz: Optional[float] = None
        self.start_freq_hz: Optional[float] = None
        self.stop_freq_hz: Optional[float] = None
        self.chirp_duration_s: Optional[float] = None
        self.frame_length: Optional[int] = None
        self.frame_duration_s: Optional[float] = None
        
        # Matplotlib
        self.fig: Optional[plt.Figure] = None
        self.pause_button: Optional[Button] = None
        self.is_paused: Callable[[], bool] = lambda: False
        
        # Keep-alive daemon
        self.keep_alive_daemon: Optional[KeepAliveDaemon] = None
        
        # FPS tracking
        self.last_fps_print: float = time.time()
        self.frames: int = 0
    
    def connect(self) -> bool:
        """
        Connect to radar system via ZMQ.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.ctx = zmq.Context()
            self.data_sock, self.data_poller = connect_sub(
                self.ctx, self.config.zmq.host, self.config.zmq.data_port
            )
            self.status_sock, self.status_poller = connect_sub(
                self.ctx, self.config.zmq.host, self.config.zmq.status_port
            )
            self.cmd_sock = HttpClient(self.config.zmq.host, self.config.zmq.command_port)
            
            # Update config with resolved host for logging
            self.config.zmq.host = self.cmd_sock.host
            
            # Get radar configuration
            self.radar_cfg = get_radar_config(self.cmd_sock)
            if not self.radar_cfg:
                self.logger.warning("Failed to retrieve radar configuration from server.")
                self.command_available = False
                return False
            
            # Extract radar parameters
            self.sample_rate_hz, self.start_freq_hz, self.stop_freq_hz, self.chirp_duration_s = extract_radar_params(self.radar_cfg)
            
            # Extract additional parameters
            fl = self.radar_cfg.get("frame_length")
            if isinstance(fl, int) and fl > 0:
                self.frame_length = fl
            
            fd = self.radar_cfg.get("frame_duration_s")
            if isinstance(fd, (int, float)) and fd > 0:
                self.frame_duration_s = float(fd)
            
            # Log extracted parameters
            if self.sample_rate_hz:
                self.logger.info(f"Sample rate: {self.sample_rate_hz/1e6:.1f} MHz")
            if self.start_freq_hz and self.stop_freq_hz:
                bandwidth_ghz = (self.stop_freq_hz - self.start_freq_hz) / 1e9
                self.logger.info(f"Bandwidth: {bandwidth_ghz:.2f} GHz")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to radar: {e}")
            self.command_available = False
            return False
    
    def start_streaming(self) -> bool:
        """
        Start radar data streaming.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.command_available or not self.cmd_sock:
            return False
        
        if not start_radar_streaming(self.cmd_sock):
            self.logger.warning("Start not acknowledged or failed.")
            try:
                self.cmd_sock.close()
            except Exception:
                pass
            self.command_available = False
            return False
        
        return True
    
    def stop_streaming(self) -> None:
        """Stop radar data streaming."""
        if self.command_available and self.cmd_sock:
            try:
                stop_radar_streaming(self.cmd_sock)
            except Exception:
                pass
    
    def setup_keep_alive(self) -> None:
        """Start keep-alive daemon."""
        if self.command_available:
            self.keep_alive_daemon = KeepAliveDaemon(
                self.config.zmq.host, self.config.zmq.command_port
            )
            self.keep_alive_daemon.start()
    
    def stop_keep_alive(self) -> None:
        """Stop keep-alive daemon."""
        if self.keep_alive_daemon:
            try:
                self.keep_alive_daemon.stop()
            except Exception:
                pass
    
    def setup_pause_button(self, position: tuple = (0.85, 0.02, 0.12, 0.04)) -> None:
        """
        Create and configure pause button.
        
        Args:
            position: Button position as [left, bottom, width, height]
        """
        if self.fig:
            self.pause_button, get_pause_state = setup_pause_button(self.fig, position, logger=self.logger)
            self.is_paused = get_pause_state
    
    @abstractmethod
    def setup_plot(self) -> None:
        """
        Setup plot-specific matplotlib figure and axes.
        
        This method should be implemented by subclasses to create
        the specific plot layout and initialize plot elements.
        """
        pass
    
    def drain_frame(self) -> Optional[Any]:
        """
        Drain and get latest frame from ZMQ socket.
        
        Can be overridden by subclasses to use drain_latest_frame_full for metadata.
        
        Returns:
            Frame data (numpy array or dict with 'data' key) or None
        """
        return drain_latest_frame(
            self.data_sock, self.data_poller, max_drain=self.config.zmq.max_drain
        )
    
    @abstractmethod
    def process_frame(self, frame_data: np.ndarray, frame_metadata: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Process a single radar frame.
        
        Args:
            frame_data: Raw radar frame data
            frame_metadata: Optional frame metadata dict (if drain_latest_frame_full was used)
            
        Returns:
            Processed data (format depends on plot type) or None to skip update
        """
        pass
    
    @abstractmethod
    def update_plot(self, processed_data: Any) -> None:
        """
        Update the plot with processed data.
        
        Args:
            processed_data: Data returned from process_frame()
        """
        pass
    
    def log_connection_info(self) -> None:
        """Log connection information."""
        self.logger.info(
            f"Subscribed data tcp://{self.config.zmq.host}:{self.config.zmq.data_port} "
            f"and status tcp://{self.config.zmq.host}:{self.config.zmq.status_port}. "
            "Close the window or Ctrl+C to quit."
        )
    
    def run(self) -> int:
        """
        Run the main plotting loop.
        
        Returns:
            Exit code (0 for success, 1 for error)
        """
        # Connect to radar
        if not self.connect():
            self.logger.error("Failed to connect to radar system.")
            return 1
        
        # Start streaming
        if not self.start_streaming():
            self.logger.warning("Failed to start streaming, continuing anyway...")
        
        # Log connection info
        self.log_connection_info()
        
        # Setup matplotlib
        plt.ion()
        self.setup_plot()
        
        # Setup pause button if figure exists
        if self.fig:
            self.setup_pause_button()
        
        # Start keep-alive daemon
        self.setup_keep_alive()
        
        try:
            while self.fig and plt.fignum_exists(self.fig.number):
                # Handle pause state
                if self.is_paused():
                    time.sleep(0.2)
                    if self.fig:
                        self.fig.canvas.flush_events()
                    continue
                
                # Drain and log latest status
                latest_status = drain_latest_status(
                    self.status_sock, self.status_poller, max_drain=5
                )
                if latest_status is not None:
                    self.logger.debug(f"STATUS: {latest_status}")
                
                # Drain and get latest frame (allow subclass to override draining method)
                latest = self.drain_frame()
                
                if latest is not None:
                    # Process frame (extract data if it's a dict, otherwise use directly)
                    if isinstance(latest, dict):
                        frame_data = latest.get('data', latest)
                    else:
                        frame_data = latest
                    
                    processed_data = self.process_frame(frame_data, frame_metadata=latest if isinstance(latest, dict) else None)
                    
                    if processed_data is not None:
                        # Update plot
                        self.update_plot(processed_data)
                        
                        # FPS tracking
                        self.frames += 1
                        now = time.time()
                        if now - self.last_fps_print >= 1.0:
                            self.logger.debug(
                                f"Update rate: ~{self.frames / (now - self.last_fps_print):.1f} Hz"
                            )
                            self.last_fps_print = now
                            self.frames = 0
                
                # Process GUI events
                plt.pause(0.001)
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
            return 0
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}", exc_info=True)
            return 1
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Cleanup resources on exit."""
        # Stop keep-alive daemon
        self.stop_keep_alive()
        
        # Stop streaming
        self.stop_streaming()
        
        # Close sockets
        if self.data_sock:
            try:
                self.data_sock.close()
            except Exception:
                pass
        
        if self.status_sock:
            try:
                self.status_sock.close()
            except Exception:
                pass
        
        if self.cmd_sock:
            try:
                self.cmd_sock.close()
            except Exception:
                pass
        
        # Terminate context
        if self.ctx:
            try:
                self.ctx.term()
            except Exception:
                pass
        
        self.logger.info("Cleanup complete")
    
    @staticmethod
    def parse_args(description: str, default_config: str = "config.yaml") -> argparse.Namespace:
        """
        Parse command-line arguments.
        
        Args:
            description: Description for argument parser
            default_config: Default config file path
            
        Returns:
            Parsed arguments
        """
        p = argparse.ArgumentParser(description=description)
        p.add_argument(
            "--config", default=default_config,
            help=f"Configuration file path (default: {default_config})"
        )
        p.add_argument(
            "--log-level",
            default="INFO",
            help="Logging level (e.g., DEBUG, INFO, WARNING). Default: INFO",
        )
        return p.parse_args()

