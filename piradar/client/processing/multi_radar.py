"""
Utilities for coordinating and fusing data from multiple radar streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Sequence

import logging
import numpy as np
import zmq

from ..communication.zmq_client import connect_sub, drain_latest_frame
from ..config import MultiRadarConfig, RadarInstanceConfig
logger = logging.getLogger(__name__)

FramePayload = Dict[str, Any]


@dataclass
class _RadarStream:
    """Internal helper tracking sockets and the last frame for a radar."""

    config: RadarInstanceConfig
    socket: zmq.Socket
    poller: zmq.Poller
    last_frame: Optional[FramePayload] = None


class MultiRadarCoordinator:
    """
    Manage multiple radar subscriptions and align frames using timestamp_ns.

    Example:
        cfg = load_config("multi.yaml", MultiRadarConfig)
        coordinator = MultiRadarCoordinator(cfg)
        frames = coordinator.get_aligned_frames()
    """

    def __init__(
        self,
        config: MultiRadarConfig,
        context: Optional[zmq.Context] = None,
    ):
        self.config = config
        self.context = context or zmq.Context.instance()
        self._owns_context = context is None
        self._streams: list[_RadarStream] = []

        for radar in self.config.enabled_radars:
            logger.info("Connecting to radar %s at %s:%s", radar.name, radar.zmq.host, radar.zmq.data_port)
            sock, poller = connect_sub(self.context, radar.zmq.host, radar.zmq.data_port)
            self._streams.append(_RadarStream(config=radar, socket=sock, poller=poller))

    def close(self) -> None:
        """Close sockets and, if owned, terminate the ZMQ context."""
        for stream in self._streams:
            try:
                stream.socket.close(0)
            except Exception:
                pass
        if self._owns_context:
            try:
                self.context.term()
            except Exception:
                pass

    def __enter__(self) -> "MultiRadarCoordinator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def get_aligned_frames(self) -> Optional[Dict[str, FramePayload]]:
        """
        Drain each radar and return aligned frames when all are within tolerance.

        Returns:
            Dict keyed by radar name -> frame payload with metadata, or None if
            frames are not yet synchronized.
        """
        frames: Dict[str, FramePayload] = {}
        for stream in self._streams:
            payload = self._drain_stream(stream)
            if payload is not None:
                frames[stream.config.name] = payload

        if len(frames) != len(self._streams):
            missing = {s.config.name for s in self._streams} - set(frames.keys())
            if missing:
                logger.debug("Waiting for frames from: %s", ", ".join(sorted(missing)))
            return None

        timestamps = [payload.get("timestamp_ns") for payload in frames.values() if payload.get("timestamp_ns") is not None]
        if timestamps:
            #print(f"max_ns - min_ns: {max(timestamps) - min(timestamps)}")
            if max(timestamps) - min(timestamps) > self.config.tolerance_ns:
                logger.debug(
                    "Frame timestamps out of tolerance (Δ=%d ns, limit=%d ns)",
                    max(timestamps) - min(timestamps),
                    self.config.tolerance_ns,
                )
                return None
        else:
            float_ts = [payload.get("timestamp") for payload in frames.values() if payload.get("timestamp") is not None]
            if float_ts:
                max_ns = int(max(float_ts) * 1e9)
                min_ns = int(min(float_ts) * 1e9)

                if max_ns - min_ns > self.config.tolerance_ns:
                    logger.debug(
                        "Float timestamps out of tolerance (Δ=%d ns, limit=%d ns)",
                        max_ns - min_ns,
                        self.config.tolerance_ns,
                    )
                    return None

        return frames

    def apply_per_radar(
        self,
        frames: Dict[str, FramePayload],
        processor: Callable[[str, FramePayload], Any],
    ) -> Dict[str, Any]:
        """
        Apply a callable to each frame payload.

        Args:
            frames: Output from get_aligned_frames().
            processor: Callable receiving (radar_name, payload) and returning a processed artifact.
        """
        return {name: processor(name, payload) for name, payload in frames.items()}

    def fuse_processed(self, processed_frames: Dict[str, np.ndarray]) -> Any:
        """
        Fuse processed outputs according to the configured strategy.

        Args:
            processed_frames: Dict of radar_name -> np.ndarray (e.g., per-radar spectra).
        """
        method = self.config.fusion.method
        if method == "per_radar":
            return processed_frames

        order: Sequence[str]
        if self.config.fusion.preserve_order:
            order = [radar.name for radar in self.config.enabled_radars if radar.name in processed_frames]
        else:
            order = sorted(processed_frames.keys())

        arrays = [processed_frames[name] for name in order]
        if not arrays:
            return processed_frames

        stacked = np.stack(arrays, axis=0)
        if method == "stack":
            return stacked
        if method == "average":
            return np.mean(stacked, axis=0)
        raise ValueError(f"Unknown fusion method '{method}'")

    def _drain_stream(self, stream: _RadarStream) -> Optional[FramePayload]:
        """Drain the latest frame for a single radar."""
        latest = drain_latest_frame(
            stream.socket,
            stream.poller,
            max_drain=stream.config.zmq.max_drain,
            include_metadata=True,
        )
        if latest is None:
            logger.debug(
                "No new frame for %s (last=%s)",
                stream.config.name,
                stream.last_frame.get("timestamp_ns") if stream.last_frame else "none",
            )
            return stream.last_frame

        if isinstance(latest, tuple) and len(latest) == 2:
            data, metadata = latest
        else:
            data, metadata = latest, {}

        payload = self._normalize_payload(stream.config, data, metadata)
        stream.last_frame = payload
        logger.debug(
            "Drained %s frame_id=%s timestamp_ns=%s",
            stream.config.name,
            payload.get("frame_id"),
            payload.get("timestamp_ns"),
        )
        return payload

    @staticmethod
    def _normalize_payload(
        radar_config: RadarInstanceConfig,
        data: np.ndarray,
        metadata: Dict[str, Any],
    ) -> FramePayload:
        """Normalize tuple returned by drain_latest_frame into a structured dict."""
        timestamp_ns = metadata.get("timestamp_ns")
        if timestamp_ns is None and metadata.get("timestamp") is not None:
            timestamp_ns = int(metadata["timestamp"] * 1e9)
        payload: FramePayload = {
            "name": radar_config.name,
            "data": data,
            "frame_id": metadata.get("frame_id"),
            "timestamp": metadata.get("timestamp"),
            "timestamp_ns": timestamp_ns,
            "metadata": {
                **radar_config.metadata,
                **{k: v for k, v in metadata.items() if k not in ("data",)},
                "radar": radar_config.name,
            },
            "calibration": radar_config.calibration,
            "color": radar_config.color,
        }
        return payload

