"""
Communication utilities for PiRadar Client.

Provides functions for connecting to radar servers, sending commands via HTTP,
and receiving data streams via ZeroMQ.
"""

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

__all__ = [
    # ZMQ utilities
    "connect_sub",
    "parse_message",
    "parse_full_message",
    "parse_status_message",
    "drain_latest_frame",
    "drain_latest_frame_full",
    "drain_latest_status",
    # HTTP client
    "HttpClient",
    "send_command",
    "get_radar_config",
    "start_radar_streaming",
    "stop_radar_streaming",
    "send_keep_alive",
    "KeepAliveDaemon",
]
