"""
DEPRECATED: This module is kept for backward compatibility only.

All functionality has been moved to:
- piradar.client.communication.http_client (HTTP commands)
- piradar.client.communication.zmq_client (ZMQ data streaming)
- piradar.client.processing (signal processing)

This module will be removed in a future version.
Please update your imports.
"""

import warnings
from typing import Optional, Tuple, Dict, Any

import numpy as np
import zmq

# Re-export from new modules with deprecation warnings
from .zmq_client import (
    connect_sub,
    parse_message,
    parse_full_message,
    parse_status_message,
    drain_latest_frame,
    drain_latest_frame_full,
    drain_latest_status,
)

from .http_client import (
    HttpClient,
    send_command,
    get_radar_config,
    start_radar_streaming,
    stop_radar_streaming,
    send_keep_alive,
    KeepAliveDaemon,
)

# Re-export signal processing functions from processing module
from ..processing import (
    apply_window_2d,
    compute_range_doppler_fft,
    calculate_range_axis_from_config,
    calculate_velocity_axis_from_config,
    process_range_doppler,
    extract_time_series,
    apply_window_1d,
    compute_fft_spectrum,
    non_coherent_integration,
    to_log_magnitude,
)

# Deprecated functions - issue warnings
def connect_req(ctx: zmq.Context, host: Optional[str], port: int) -> zmq.Socket:
    """
    DEPRECATED: This function is deprecated and will be removed.
    Use HttpClient for commands instead.
    """
    warnings.warn(
        "connect_req is deprecated and will be removed. Use HttpClient for commands.",
        DeprecationWarning,
        stacklevel=2
    )
    from .common import resolve_host
    host = resolve_host(host)
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.setsockopt(zmq.SNDTIMEO, 2000)
    sock.setsockopt(zmq.LINGER, 0)
    return sock


class RobustRequestClient:
    """
    DEPRECATED: This class is deprecated and will be removed.
    Use HttpClient instead.
    """
    def __init__(self, ctx: zmq.Context, host: Optional[str], port: int, 
                 timeout_ms: int = 2000, retries: int = 3):
        warnings.warn(
            "RobustRequestClient is deprecated and will be removed. Use HttpClient instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.ctx = ctx
        from .common import resolve_host
        self.host = resolve_host(host)
        self.port = port
        self.timeout_ms = timeout_ms
        self.retries = retries
        self.socket = connect_req(ctx, host, port)
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

    def send_command(self, command_id: str, parameters: Optional[dict] = None) -> Optional[dict]:
        warnings.warn(
            "RobustRequestClient.send_command is deprecated and non-functional. Use HttpClient.",
            DeprecationWarning,
            stacklevel=2
        )
        return None
    
    def close(self):
        if self.socket:
            self.socket.close()


# Issue deprecation warning when module is imported
warnings.warn(
    "piradar.client.communication.radar_plot_helpers is deprecated. "
    "Import from piradar.client.communication (http_client, zmq_client) "
    "and piradar.client.processing instead.",
    DeprecationWarning,
    stacklevel=2
)
