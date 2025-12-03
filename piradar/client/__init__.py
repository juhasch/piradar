"""
PiRadar Client Package

A Python package for radar signal processing and visualization
for remote radar systems connected via ZeroMQ.

This package provides:
- ZMQ communication utilities
- Radar signal processing functions
- Configuration management
- Plotting utilities
"""

__version__ = "0.1.0"
__author__ = "PiRadar Team"

from .communication import *
from .config import *
from .processing import *
from .plotting import *



