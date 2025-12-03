"""
Live velocity waterfall plot from ZeroMQ.

Subscribes to tcp://<host>:<port>, computes range–Doppler, then reduces
each frame to a velocity vector by taking the max over all range bins per
velocity bin. Displays a scrolling waterfall over time with:
- X axis: Velocity (m/s)
- Y axis: Frame index (newest at bottom)
- Color: Magnitude (dB)

Usage:
  python zmq_plot_velocity_waterfall.py --config config/velocity_waterfall.yaml

Requirements:
  pip install pyzmq matplotlib numpy scipy pydantic pyyaml
"""

import argparse
import logging
import sys
import time
from typing import Optional

import numpy as np
import zmq
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt backend for interactive performance
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from piradar.client.config import VelocityWaterfallConfig, load_config
from piradar.client.logging_utils import configure_logging
from piradar.client.communication import (
    connect_sub, drain_latest_frame, drain_latest_status,
    start_radar_streaming, stop_radar_streaming, get_radar_config,
    send_command, HttpClient, KeepAliveDaemon
)

from piradar.client.processing import (
    calculate_velocity_axis_from_config,
    process_range_doppler
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live velocity waterfall from ZMQ range–Doppler frames")
    p.add_argument("--config", default="velocity_waterfall.yaml", 
                   help="Configuration file path (default: velocity_waterfall.yaml)")
    p.add_argument("--log-level", default="INFO",
                   help="Logging level (e.g., DEBUG, INFO, WARNING). Default: INFO")
    return p.parse_args()



def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Load configuration
    try:
        config = load_config(args.config, VelocityWaterfallConfig)
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        return 1

    ctx = zmq.Context()
    data_sock, data_poller = connect_sub(ctx, config.zmq.host, config.zmq.data_port)
    status_sock, status_poller = connect_sub(ctx, config.zmq.host, config.zmq.status_port)
    # Use HttpClient for command channel (HTTP API)
    cmd_sock = HttpClient(config.zmq.host, config.zmq.command_port)
    # Update config with resolved host for logging
    config.zmq.host = cmd_sock.host
    command_available = True

    # Query radar configuration
    start_freq_hz: Optional[float] = None
    stop_freq_hz: Optional[float] = None
    frame_length: Optional[int] = None
    frame_duration_s: Optional[float] = None

    radar_cfg = get_radar_config(cmd_sock)
    if radar_cfg:
        sf = radar_cfg.get("start_frequency_Hz")
        if isinstance(sf, (int, float)) and sf > 0:
            start_freq_hz = float(sf)
        ef = radar_cfg.get("stop_frequency_Hz")
        if isinstance(ef, (int, float)) and ef > 0:
            stop_freq_hz = float(ef)
        fl = radar_cfg.get("frame_length")
        if isinstance(fl, int) and fl > 0:
            frame_length = fl
        fd = radar_cfg.get("frame_duration_s")
        if isinstance(fd, (int, float)) and fd > 0:
            frame_duration_s = float(fd)
    else:
        logging.warning("Failed to retrieve initial status from server.")
        try:
            cmd_sock.close()
        except Exception:
            pass
        command_available = False

    # Try to start streaming
    if command_available:
        if not start_radar_streaming(cmd_sock):
            try:
                cmd_sock.close()
            except Exception:
                pass
            command_available = False

    logging.info("Subscribed data tcp://%s:%s and status tcp://%s:%s", config.zmq.host, config.zmq.data_port, config.zmq.host, config.zmq.status_port)
    logging.info("Window: %s, Colormap: %s, History: %d", config.plot.window, config.plot.colormap, config.history)

    # Matplotlib setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Velocity Waterfall (max over range bins)")
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Frame index (newest at bottom)")

    # Initialize with a small placeholder; will resize on first frame
    history = int(config.history)
    placeholder = np.zeros((history, 10), dtype=float)
    im = ax.imshow(placeholder, cmap=config.plot.colormap, aspect='auto', origin='lower')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude (dB)')

    # Add pause button
    button_ax = plt.axes([0.85, 0.02, 0.12, 0.04])
    pause_button = Button(button_ax, 'Pause', color='lightblue', hovercolor='lightcyan')

    # Pause state tracking
    is_paused = False

    def toggle_pause(event):
        nonlocal is_paused
        is_paused = not is_paused
        if is_paused:
            pause_button.label.set_text('Resume')
            pause_button.color = 'lightcoral'
            pause_button.hovercolor = 'lightpink'
            logging.info("Data updates paused")
        else:
            pause_button.label.set_text('Pause')
            pause_button.color = 'lightblue'
            pause_button.hovercolor = 'lightcyan'
            logging.info("Data updates resumed")
        pause_button.ax.figure.canvas.draw_idle()

    pause_button.on_clicked(toggle_pause)
    
    # Start keep-alive daemon
    keep_alive_daemon = KeepAliveDaemon(config.zmq.host, config.zmq.command_port)
    keep_alive_daemon.start()

    last_fps_print = time.time()
    frames = 0
    waterfall: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    baseline: Optional[np.ndarray] = None

    try:
        while plt.fignum_exists(fig.number):
            if is_paused:
                time.sleep(0.2)
                fig.canvas.flush_events()
                continue
            # Drain and log status
            latest_status = drain_latest_status(status_sock, status_poller, max_drain=5)
            if latest_status is not None:
                logging.debug("STATUS: %s", latest_status)

            latest = drain_latest_frame(data_sock, data_poller, max_drain=config.zmq.max_drain)
            if latest is not None:
                # Optional static clutter suppression via EWMA baseline subtraction
                ts = latest
                if config.suppress_static:
                    if (baseline is None) or (baseline.shape != ts.shape):
                        baseline = ts.astype(np.float64).copy()
                    alpha = float(config.suppress_alpha)
                    # Clamp alpha to a reasonable range
                    if alpha < 0.0:
                        alpha = 0.0
                    if alpha >= 1.0:
                        alpha = 0.9999
                    baseline = alpha * baseline + (1.0 - alpha) * ts
                    ts = ts - baseline

                rd_map = process_range_doppler(ts, config.plot.window, config.plot.range_bins,
                                                config.plot.doppler_bins, config.plot.db_min, config.plot.db_max)

                # Optionally use only positive range bins
                if config.positive_ranges_only:
                    num_range_bins = rd_map.shape[1]
                    positive_range_bins = num_range_bins // 2
                    rd_used = rd_map[:, positive_range_bins:]
                else:
                    rd_used = rd_map

                # Ignore the first range bin to reduce close-in leakage/clutter
                if rd_used.shape[1] > 1:
                    rd_used = rd_used[:, 68:]

                # Reduce to velocity vector by max over range axis
                # rd_used shape: (num_doppler_bins, num_range_bins_used)
                velocity_vector = np.max(rd_used, axis=1)

                # Compute velocity axis (from config if available)
                num_doppler_bins = rd_map.shape[0]
                if (
                    isinstance(frame_length, int) and frame_length > 0 and
                    isinstance(frame_duration_s, float) and frame_duration_s > 0.0 and
                    isinstance(start_freq_hz, float) and isinstance(stop_freq_hz, float)
                ):
                    center_freq_hz = 0.5 * (start_freq_hz + stop_freq_hz)
                    velocities = calculate_velocity_axis_from_config(num_doppler_bins, frame_length, frame_duration_s, center_freq_hz)
                else:
                    # Fallback - use default values if config not available
                    logging.warning("Radar configuration not available, using default velocity axis")
                    velocities = np.linspace(-5, 5, num_doppler_bins)  # ±5 m/s range as fallback

                # Initialize or validate waterfall buffer
                if (waterfall is None) or (waterfall.shape[1] != velocity_vector.shape[0]):
                    waterfall = np.zeros((history, velocity_vector.shape[0]), dtype=float)
                    im.set_data(waterfall)
                    extent = [float(velocities[0]), float(velocities[-1]), 0.0, float(history)]
                    im.set_extent(extent)
                    ax.set_xlim(velocities[0], velocities[-1])
                    ax.set_ylim(0, history)

                # Scroll up and append new row at bottom
                waterfall = np.roll(waterfall, -1, axis=0)
                waterfall[-1, :] = velocity_vector

                # Update image and color scaling
                current_max = float(np.max(waterfall))
                vmin = 50 #current_max - float(config.plot.db_range)
                vmax = 90 # current_max
                im.set_data(waterfall)
                im.set_clim(vmin=config.plot.db_min, vmax=config.plot.db_max)

                # Efficient redraw
                fig.canvas.draw_idle()

                frames += 1
                now = time.time()
                if now - last_fps_print >= 1.0:
                    logging.debug("Update rate: ~%.1f Hz", frames / (now - last_fps_print))
                    last_fps_print = now
                    frames = 0

            plt.pause(0.001)

        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        keep_alive_daemon.stop()

        if command_available:
            stop_radar_streaming(cmd_sock)
        try:
            data_sock.close()
            status_sock.close()
        except Exception:
            pass
        try:
            cmd_sock.close()
        except Exception:
            pass
        ctx.term()


if __name__ == "__main__":
    sys.exit(main())


