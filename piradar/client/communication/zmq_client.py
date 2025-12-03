"""
ZeroMQ client utilities for radar data streaming.

Provides functions for connecting to ZMQ data streams, parsing messages,
and draining frames/status updates.
"""

import logging
import pickle
from typing import Optional, Tuple, Dict, Any, Union

import numpy as np
import zmq

from .common import resolve_host


def connect_sub(ctx: zmq.Context, host: Optional[str], port: int) -> Tuple[zmq.Socket, zmq.Poller]:
    """Create and configure a ZMQ SUB socket for data/status subscription."""
    host = resolve_host(host)
    sock = ctx.socket(zmq.SUB)
    sock.connect(f"tcp://{host}:{port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    # High water mark to avoid unbounded memory growth
    sock.setsockopt(zmq.RCVHWM, 10_000)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    return sock, poller


def parse_message(msg: bytes, include_metadata: bool = False) -> Optional[
    Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]
]:
    """Parse ZMQ message and return numpy array or (array, metadata).

    Args:
        msg: Pickled message payload.
        include_metadata: When True, return a tuple ``(data, metadata)``, where
            metadata is a dictionary containing frame_id/timestamps/etc.
    """
    try:
        obj = pickle.loads(msg)
    except Exception as exc:
        logging.debug("Failed to unpickle message: %s", exc)
        return None

    # Simple message format: {"frame_id": int, "timestamp": float, "data": array}
    if isinstance(obj, dict) and "data" in obj:
        data = obj["data"]
        if include_metadata:
            metadata = {k: v for k, v in obj.items() if k != "data"}
            return data, metadata
        return data

    # Raw array
    if isinstance(obj, np.ndarray):
        return obj

    return None


def parse_full_message(msg: bytes) -> Optional[Dict[str, Any]]:
    """Parse ZMQ message and return full message dict with frame_id, timestamp, and data.
    
    Returns:
        Dict with keys 'frame_id', 'timestamp', 'data' if message is in dict format,
        or None if parsing fails or message is in raw array format.
    """
    try:
        obj = pickle.loads(msg)
    except Exception as exc:
        logging.debug("Failed to unpickle message: %s", exc)
        return None

    # Full message format: {"frame_id": int, "timestamp": float, "data": array}
    if isinstance(obj, dict) and "data" in obj and "frame_id" in obj and "timestamp" in obj:
        return obj

    # If it's just an array, we can't return frame_id/timestamp
    return None


def parse_status_message(msg: bytes) -> Optional[str]:
    """Parse status message into a concise string for logging."""
    try:
        obj = pickle.loads(msg)
        if isinstance(obj, dict):
            parts = []
            for key in ("radar_status", "frame_count", "fps", "uptime"):
                if key in obj:
                    parts.append(f"{key}={obj[key]}")
            if parts:
                return ", ".join(parts)
            return str(obj)
        return str(obj)
    except Exception:
        try:
            return msg.decode("utf-8", errors="replace")
        except Exception:
            return f"<status bytes {len(msg)} B>"


def drain_latest_frame(
    sock: zmq.Socket,
    poller: zmq.Poller,
    max_drain: int = 50,
    include_metadata: bool = False,
) -> Optional[Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]]:
    """Drain up to max_drain messages and return the latest valid frame array.
    
    Args:
        sock: The SUB socket to drain from.
        poller: The poller for the SUB socket.
        max_drain: Maximum number of messages to drain.
        include_metadata: When True, return tuple (data, metadata).
    """
    latest = None
    drained = 0
    while drained < max_drain:
        events = dict(poller.poll(timeout=0))
        if sock not in events:
            break
        try:
            msg = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        arr = parse_message(msg, include_metadata=include_metadata)
        if include_metadata:
            if isinstance(arr, tuple) and len(arr) == 2 and isinstance(arr[0], np.ndarray):
                latest = arr
        else:
            if isinstance(arr, np.ndarray):
                latest = arr
        drained += 1
    return latest


def drain_latest_frame_full(sock: zmq.Socket, poller: zmq.Poller, max_drain: int = 50) -> Optional[Dict[str, Any]]:
    """Drain up to max_drain messages and return the latest valid frame with full metadata.
    
    Args:
        sock: The SUB socket to drain from.
        poller: The poller for the SUB socket.
        max_drain: Maximum number of messages to drain.
    
    Returns:
        Dict with keys 'frame_id', 'timestamp', 'data' or None if no valid frame found.
    """
    latest: Optional[Dict[str, Any]] = None
    drained = 0
    while drained < max_drain:
        events = dict(poller.poll(timeout=0))
        if sock not in events:
            break
        try:
            msg = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        frame_dict = parse_full_message(msg)
        if isinstance(frame_dict, dict) and "data" in frame_dict:
            latest = frame_dict
        drained += 1
    return latest


def drain_latest_status(sock: zmq.Socket, poller: zmq.Poller, max_drain: int = 20) -> Optional[str]:
    """Drain up to max_drain status messages and return the latest one."""
    latest: Optional[str] = None
    drained = 0
    while drained < max_drain:
        events = dict(poller.poll(timeout=0))
        if sock not in events:
            break
        try:
            msg = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        latest = parse_status_message(msg)
        drained += 1
    return latest

