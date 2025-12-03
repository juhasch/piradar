"""
Common utilities for radar communication.

Provides shared functions for host resolution and discovery.
"""

import logging
from typing import Optional

from piradar.hw.discovery import RadarScanner

_cached_host = None


def resolve_host(host: Optional[str]) -> str:
    """Resolve ZMQ host. If host is None, try to discover via Zeroconf.
    Uses caching to avoid multiple scans in the same process."""
    global _cached_host
    
    if host:
        return host
        
    if _cached_host:
        return _cached_host
    
    logging.info("No host specified. Scanning for radars on network...")
    scanner = RadarScanner()
    radars = scanner.scan(timeout=2.0)
    
    if not radars:
        logging.warning("No radars found via discovery. Defaulting to localhost.")
        _cached_host = "localhost"
        return "localhost"
    
    # Pick the first one
    radar = radars[0]
    logging.info(f"Auto-selected radar: {radar.name} at {radar.address}")
    _cached_host = radar.address
    return radar.address

