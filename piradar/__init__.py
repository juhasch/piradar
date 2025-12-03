"""
PiRadar - Pythonic BGT60TR13C radar sensor driver
"""

# Check if hardware is available
_hardware_available = False

try:
    # Try to import eeprom module - this will fail if hardware dependencies are missing
    from .hw.eeprom import check_hat
    
    # Verify actual hardware is present
    check_hat()
    
    # Hardware is available - import all hardware modules
    _hardware_available = True
    from .hw import (
        BGT60TR13C, 
        BGT60TR13CError, 
        ChipIDError,
        AsyncZMQHandler, 
        AsyncZMQClient, 
        ZMQHandler, 
        ZMQClient,
        SpiInterface,
        check_hat,
        read_uint12,
        AdaptiveRadarController,
        create_adaptive_controller,
    )
except (ImportError, RuntimeError, Exception):
    # Hardware not available - set all to None
    BGT60TR13C = None
    BGT60TR13CError = None
    ChipIDError = None
    AsyncZMQHandler = None
    AsyncZMQClient = None
    ZMQHandler = None
    ZMQClient = None
    SpiInterface = None
    check_hat = None
    read_uint12 = None
    AdaptiveRadarController = None
    create_adaptive_controller = None

__all__ = [
    "AsyncZMQHandler",
    "AsyncZMQClient", 
    "ZMQHandler",
    "ZMQClient",
    "BGT60TR13C",
    "BGT60TR13CError",
    "ChipIDError",
    "SpiInterface",
    "check_hat",
    "read_uint12",
    "AdaptiveRadarController",
    "create_adaptive_controller",
]

__version__ = "0.1.0"

