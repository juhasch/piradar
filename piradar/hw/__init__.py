"""
PiRadar Hardware Package

Hardware access components for the BGT60TR13C radar sensor.
Requires hardware-specific dependencies (gpiod, spidev, smbus2).

Install with: pip install piradar
"""

from .bgt60tr13c import BGT60TR13C, BGT60TR13CError, ChipIDError
from .zmq_handler import AsyncZMQHandler, AsyncZMQClient, ZMQHandler, ZMQClient, RadarStreamer
from .spiinterface import SpiInterface
from .eeprom import check_hat
from .utility import read_uint12
from .radar_controller import AdaptiveRadarController, create_adaptive_controller
from .server import RadarServer

__all__ = [
    "BGT60TR13C",
    "BGT60TR13CError",
    "ChipIDError",
    "AsyncZMQHandler",
    "RadarStreamer",
    "AsyncZMQClient",
    "ZMQHandler",
    "ZMQClient",
    "SpiInterface",
    "check_hat",
    "read_uint12",
    "AdaptiveRadarController",
    "create_adaptive_controller",
    "RadarServer",
]
