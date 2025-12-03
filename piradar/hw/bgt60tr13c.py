"""
Driver for the BGT60TR13C radar sensor.
"""
import os
import logging
import time
import numpy as np
import queue
import threading
from typing import Optional, List
from dataclasses import dataclass

import gpiod
from gpiod.line import Edge, Direction, Value

from .utility import read_uint12
from ..config import RadarConfig
from .eeprom import check_hat
from ..registermap import RegisterMap
from .spiinterface import SpiInterface
from ..registers import read_registers
from .default_config import default_config

BGT60TRXX_REG_FSTAT_TR13C_FIFO_SIZE = 8192

logger = logging.getLogger(__name__)
class BGT60TR13CError(Exception):
    """Base exception for BGT60TR13C errors."""
    pass

class ChipIDError(BGT60TR13CError):
    """Raised when chip ID verification fails."""
    pass

class FIFOError(BGT60TR13CError):
    """Raised when FIFO operations fail."""
    pass

class SoftResetError(BGT60TR13CError):
    """Raised when soft reset fails."""
    pass

class FIFOParameterError(BGT60TR13CError):
    """Raised when FIFO parameters are invalid."""
    pass

@dataclass
class FifoStatus:
    """Data class to hold FIFO status information."""
    fof_err: int
    full: int
    cref: int
    empty: int
    fuf_err: int
    spi_burst_err: int
    clk_num_err: int
    fill_status: int


class BGT60TR13C:
    """
    BGT60TR13C radar sensor driver
    
    This class provides a Pythonic interface to the BGT60TR13C radar sensor,
    handling SPI communication, GPIO control, and data collection.
    """
    running: bool = False
    num_tx_channels: int = 1
    num_rx_channels: int = 3
    f_sysclk = 80_000_000
    t_sysclk_ns = 1/f_sysclk * 1e9
    
    RESET_SW = 0x2
    RESET_FSM = 0x4
    RESET_FIFO = 0x8
    
    def __init__(self, 
                 spi_bus: int = 0, 
                 spi_dev: int = 0, 
                 spi_speed: int = 40_000_000,
                 rst_pin: int = 12, 
                 irq_pin: int = 25,
                 check_hardware: bool = True):
        """
        Initialize the BGT60TR13C radar sensor.
        
        Args:
            spi_bus: SPI bus number
            spi_dev: SPI device number
            spi_speed: SPI speed in Hz
            rst_pin: GPIO pin for reset signal
            irq_pin: GPIO pin for interrupt signal
        """        
        if check_hardware:
            check_hat()  # First check if hat is actually installed
        
        self._spi = SpiInterface(spi_bus, spi_dev, spi_speed)
        
        gpio_config = {
            rst_pin: gpiod.LineSettings(
                direction=Direction.OUTPUT,
                output_value=Value.ACTIVE
            ),
            irq_pin: gpiod.LineSettings(
                direction=Direction.INPUT,
                edge_detection=Edge.RISING
            )
        }
        
        self._gpio_request = gpiod.request_lines(
            path="/dev/gpiochip0",
            config=gpio_config,
            consumer="BGT60TR13C"
        )
        
        # Load the register map
        import importlib.resources
        registermap_path = importlib.resources.files("piradar") / "bgt60tr13c_registermap.yaml"
        logger.debug(f"Loading register map from package resources")
        self.registermap = RegisterMap(str(registermap_path), self._spi)

        self._rst_pin_offset = rst_pin
        self._irq_pin_offset = irq_pin
        
        # Data buffers
        self.frame_buffer: queue.Queue = queue.Queue(maxsize=256)
        self._sub_frame_buffer: List[int] = []
                
        # Data collection thread
        self._data_collection_thread: Optional[threading.Thread] = None
        self._data_collection_stop_event = threading.Event()
        # Cache for properties that would otherwise trigger SPI reads
        self._frame_length_cache: Optional[int] = None

    @property
    def is_running(self) -> bool:
        """Return True if the data collection thread is running."""
        return bool(self._data_collection_thread and self._data_collection_thread.is_alive())

    # Minimal public register accessors to support adaptive control / RL
    def read_register(self, address: int) -> int:
        """Read a raw register value via SPI (thread-safe)."""
        return self._spi.read_register(address)

    def write_register(self, address: int, value: int) -> None:
        """Write a raw register value via SPI (thread-safe)."""
        self._spi.write_register_with_speed_optimization(address, value)

    def apply_config(self, config: dict) -> None:
        """
        Apply a dictionary-based configuration using the RegisterMap.
        
        Args:
            config: Dictionary where keys are 'REGISTER.FIELD', 'REGISTER' (str),
                   or register address (int), and values are integers.
        """
        # First, reset all registers to default in the map to ensure clean state
        # (Optional, but good practice if we assume we start from reset state)
        # self.registermap.reset_all_registers() 
        
        for key, value in config.items():
            if isinstance(key, str) and '.' in key:
                # Set field
                try:
                    self.registermap.set_field(key, value)
                    logger.debug(f"Set field {key} = {value}")
                except Exception as e:
                    logger.warning(f"Failed to set field {key}: {e}")
            elif isinstance(key, int):
                # Set register by address
                try:
                    self.write_register(key, value)
                    logger.debug(f"Set register address 0x{key:02X} = 0x{value:X}")
                except Exception as e:
                    logger.warning(f"Failed to set register address 0x{key:02X}: {e}")
            else:
                # Set full register by name
                try:
                    reg = self.registermap.get_register(key)
                    if reg:
                        reg.value = value
                        logger.debug(f"Set register {key} = 0x{value:X}")
                    else:
                        logger.warning(f"Unknown register {key}")
                except Exception as e:
                    logger.warning(f"Failed to set register {key}: {e}")

    def legacy_configure(self, register_filename: str) -> None:
        """
        Configure the radar from a register text file.
        
        Args:
            register_filename: Path to register configuration file
        """
        self._registers = read_registers(register_filename)
        self._apply_register_config()

    def configure(self, radar_config: RadarConfig | None = None) -> None:
        """
        Configure the radar based on the config class
        
        Args:
            radar_config: Optional RadarConfig object with custom configuration
        """
        logger.debug("Performing hard reset before configuration...")
        self.hard_reset()
        # Wait a bit after hard reset for chip to settle
        time.sleep(0.05)
        
        try:
            self.check_chip_id()
        except Exception as e:
            logger.warning(f"Pre-check failed: {e}")

        logger.debug("Applying default configuration...")
        # Apply default configuration first
        self.apply_config(default_config)
        
        if radar_config is not None:
            self._radar_config = radar_config
            try:
                self.set_start_frequency(radar_config.start_frequency_Hz)
            except ValueError as e:
                logger.error(f"Configuration Error: {e}")
                # Try to recover or debug by printing registers related to freq
                divset = self.registermap.PACR2.DIVSET.value
                logger.error(f"Debug: DIVSET={divset}")
                raise e

            self.set_stop_frequency(radar_config.stop_frequency_Hz)
            self.set_chirp_duration(radar_config.chirp_duration_s)
            self.frame_length = radar_config.frame_length
            self.frame_duration = radar_config.frame_duration_s
            self.output_power = radar_config.output_power
            self.adc_sample_rate = radar_config.adc_sample_rate_Hz

        
        # Also read ADC_DIV for logging purposes
        adc_div_value = self.registermap.ADC0.ADC_DIV.value
        adc_sampling_freq = self.f_sysclk / adc_div_value
        
        # Read RTU1 to calculate actual chirp duration for validation
        rtu1_value = self.registermap.PLL1_2.RTU1.value
        # RTU defines the number of clock cycles for the Upchirp
        chirp_duration_ns = rtu1_value * 8 * self.t_sysclk_ns
        chirp_duration_us = chirp_duration_ns / 1000  # microseconds
        
        # Validate that the number of samples makes sense for the chirp duration
        expected_samples = int(chirp_duration_ns * adc_sampling_freq / 1e9)
        
        logger.debug(f"ADC_DIV register value: {adc_div_value}")
        logger.debug(f"ADC sampling frequency: {adc_sampling_freq / 1e6:.2f} MHz")
        logger.debug(f"RTU1 register value: {rtu1_value}")
        logger.debug(f"Calculated chirp duration: {chirp_duration_us:.2f} µs")
        logger.debug(f"Expected samples for chirp duration: {expected_samples}")
        
        logger.debug(f"num_rx_channels: {self.num_rx_channels}")
        logger.debug(f"num_tx_channels: {self.num_tx_channels}")
        logger.debug(f"frame_duration: {self.frame_duration*1e3:.2f} ms")
        logger.debug(f"k_chirp: {self.k_chirp():.2f} Hz/s")
        self.set_fifo_parameters()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self._cleanup_gpio()
    
    def _cleanup_gpio(self):
        """Clean up GPIO resources."""
        if hasattr(self, '_gpio_request'):
            self._gpio_request.release()
       
    
    def _set_default_register(self, reg_addr: int, data: int) -> None:
        """Set the registers to the default values."""
        for reg_addr, reg in self._registers.items():
            if reg.default_value is not None:
                self._spi.write_register_with_speed_optimization(reg_addr, reg.default_value)
        
    def set_fifo_parameters(self) -> None:
        """
        Set FIFO parameters for data collection.
        
        Raises:
            FIFOParameterError: If parameters are invalid
        """
        max_fifo_size = BGT60TRXX_REG_FSTAT_TR13C_FIFO_SIZE
        
        # Check buffer size of spidev, up to 65536
        with open('/sys/module/spidev/parameters/bufsiz', 'r') as file:
            max_buffer_size = int(file.read())

        if max_buffer_size >= max_fifo_size:
            num_samples_irq = max_fifo_size
        else:
            num_samples_irq = max_buffer_size

        self._set_fifo_limit(num_samples_irq)
        
    def check_chip_id(self) -> None:
        """
        Check and verify the chip ID.
        
        Raises:
            ChipIDError: If the chip is not BGT60TR13C
        """
        # Use the registermap interface to get chip ID fields
        chip_id_reg = self.registermap.CHIP_ID
        logger.debug(f"CHIP_ID: 0x{chip_id_reg.value:08X}")
        
        chip_id_digital = chip_id_reg.DIGITAL_ID.value
        chip_id_rf = chip_id_reg.RF_ID.value
        
        if chip_id_digital == 3 and chip_id_rf == 3:
            logger.debug("Chip verified as BGT60TR13C")
        else:
            raise ChipIDError("Chip is NOT BGT60TR13C")
    
    def _apply_register_config(self) -> None:
        """Apply register configuration. This should eventually be replaced using the RegisterMap class instead."""
        for address in self._registers:
            address = self._registers[address].addr
            value = self._registers[address].default_value

            if value is not None:
                logger.debug(f"Setting register {address} to {value}")
                self._spi.write_register_with_speed_optimization(address, value)

    def start(self) -> None:
        """Start data collection thread."""
        if self._data_collection_thread and self._data_collection_thread.is_alive():
            self.stop()
        
        self.running = True
        self._data_collection_stop_event.clear()
        self._data_collection_thread = threading.Thread(target=self._data_collection, daemon=True)
        self._data_collection_thread.start()
    
    def stop(self) -> None:
        """
        Stop data collection and cleanup.
        
        Raises:
            SoftResetError: If soft reset fails
        """
        self.running = False
        self._spi.stop_frame()

        if self._data_collection_thread and self._data_collection_thread.is_alive():
            self._data_collection_stop_event.set()
            self._data_collection_thread.join(timeout=1)

        self.soft_reset(self.RESET_FSM)       
        self.soft_reset(self.RESET_FIFO)       

                                
    def soft_reset(self, reset_type: int) -> None:
        """
        Perform soft reset.
        
        Args:
            reset_type: Type of reset to perform
            
        Raises:
            SoftResetError: If soft reset times out
        """
        self._sub_frame_buffer.clear()
        
        # Use the registermap interface to set reset bits
        main_reg = self.registermap.MAIN
        
        # Determine which reset fields to set based on reset_type
        if reset_type & self.RESET_SW:  # Software reset
            main_reg.SW_RESET.value = 1
        if reset_type & self.RESET_FSM:  # FSM reset  
            main_reg.FSM_RESET.value = 1
        if reset_type & self.RESET_FIFO:  # FIFO reset
            main_reg.FIFO_RESET.value = 1
        
        # Wait for reset to complete
        timeout = 0.1  # 100ms timeout
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            time.sleep(0.01)
            
            # Check if reset bits have cleared (W1C fields auto-clear)
            current_status = main_reg.value
            if not (current_status & reset_type):
                time.sleep(0.01)
                return
        
        logger.error("Soft reset timeout")
        raise SoftResetError("Soft reset timeout")
    
    def hard_reset(self) -> None:
        """Perform hard reset using GPIO."""
        self._gpio_request.set_value(self._rst_pin_offset, Value.INACTIVE)
        time.sleep(0.01)
        self._gpio_request.set_value(self._rst_pin_offset, Value.ACTIVE)
    
    def _check_fifo_status(self) -> FifoStatus:
        """Check FIFO status and return structured data."""
        # Use the registermap interface to get individual field values
        fstat = self.registermap.FSTAT
        
        return FifoStatus(
            fof_err=fstat.FOF_ERR.value,
            full=fstat.FULL.value,
            cref=fstat.CREF.value,
            empty=fstat.EMPTY.value,
            fuf_err=fstat.FUF_ERR.value,
            spi_burst_err=fstat.SPI_BURST_ERR.value,
            clk_num_err=fstat.CLK_NUM_ERR.value,
            fill_status=fstat.FILL_STATUS.value
        )
    
    def _set_fifo_limit(self, num_samples: int) -> None:
        """
        Set FIFO limit.
        
        Args:
            num_samples: Number of samples for FIFO limit
            
        Raises:
            FIFOParameterError: If num_samples is invalid
        """
        if not (0 < (num_samples >> 1) <= BGT60TRXX_REG_FSTAT_TR13C_FIFO_SIZE and num_samples % 2 == 0):
            raise FIFOParameterError("Invalid num_samples for FIFO limit")
        
        # Use the registermap interface to set the FIFO_CREF field
        sfctl = self.registermap.SFCTL
        fifo_cref_value = (num_samples >> 1) - 1
        sfctl.FIFO_CREF.value = fifo_cref_value
        
        logger.debug(f"Setting FIFO limit to: {hex(sfctl.value)}, num_samples: {num_samples}, FIFO_CREF: {fifo_cref_value}")
    
    def _print_fifo_status(self) -> None:
        """Print FIFO status information."""
        status = self._check_fifo_status()
        logger.debug(f"FIFO Status - FOF_ERR: {status.fof_err}, FUF_ERR: {status.fuf_err}, "
                    f"SPI_BURST_ERR: {status.spi_burst_err}, CLK_NUM_ERR: {status.clk_num_err}")    
    
    def _data_collection(self) -> None:
        """Main data collection loop with interrupt-driven FIFO reading for 50 MHz SPI."""
        logger.debug("Data collection thread started")
        
        # We need to load the read_uint12 function once to warm up the JIT compiler
        test_data = bytes([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB])
        read_uint12(test_data)
        logger.debug(f"samples_per_chirp: {self.num_samples_per_chirp}")
        self.num_samples_per_frame = self.num_samples_per_chirp * self.num_rx_channels * self.num_tx_channels * self.frame_length
        logger.debug(f"num_samples_per_frame: {self.num_samples_per_frame}")
        self._num_bytes_per_frame = self.num_samples_per_frame * 12 // 8

        self.soft_reset(self.RESET_FSM)       
        self.soft_reset(self.RESET_FIFO)       

        while not self._data_collection_stop_event.is_set():
            self._spi.start_frame()
            logger.debug("Frame started - waiting for interrupts...")
            interrupt_count = 0
            
            logger.debug(f"Bytes per frame: {self._num_bytes_per_frame}")

            while not self._data_collection_stop_event.is_set():
                # Wait for edge events (interrupts) using proper gpiod mechanism
                events = self._gpio_request.read_edge_events()
                if events:
                    for event in events:
                        interrupt_count += 1
                        
                        # Read all available data from FIFO - we got an interrupt, so data is there
                        fifo_status = self._check_fifo_status()
                        logger.debug(f"FIFO status: {fifo_status}")
                        
                        if fifo_status.cref > 0:
                            samples_to_read = (fifo_status.fill_status >> 2) * 4
                            logger.debug(f"samples_to_read: {samples_to_read}")
                            fifo_data = self._spi._get_fifo_data(samples_to_read)
                            
                            # Process data immediately
                            self._sub_frame_buffer.extend(fifo_data)
                            
                            # Process complete frames as soon as they're available
                            while len(self._sub_frame_buffer) >= self._num_bytes_per_frame:
                                frame_data = self._sub_frame_buffer[:self._num_bytes_per_frame]
                                self._sub_frame_buffer = self._sub_frame_buffer[self._num_bytes_per_frame:]
                                
                                full_frame = bytes(frame_data)
                                
                                # Add to frame buffer
                                try:
                                    frame = read_uint12(full_frame).reshape(self.frame_length, self.num_samples_per_chirp, self.num_rx_channels)
                                    self.frame_buffer.put(frame, block=False)
                                    logger.debug(f"Frame added to buffer, total frames: {self.frame_buffer.qsize()}")
                                except queue.Full:
                                    logger.warning("Frame buffer is full, dropping frame")
                else:
                    time.sleep(0.001)
            logger.debug(f"Data collection loop ended. Total interrupts received: {interrupt_count}")
                        
        logger.debug("Data collection thread stopped")

    def get_frame(self) -> tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Get the latest radar frame from the buffer.
        
        Returns:
            Tuple containing (frame_data, metadata)
            frame_data: Numpy array of shape (num_chirps, samples_per_chirp, num_rx)
            metadata: Dictionary with frame metadata or None
        """
        try:
            # Return None if no frame is available immediately
            if self.frame_buffer.empty():
                return None, None
                
            frame = self.frame_buffer.get_nowait()
            
            metadata = {
                "timestamp": time.time(),
                "sequence_index": 0, # Default sequence
                "frame_length": self.frame_length,
                "num_chirps": self.frame_length,
                "samples_per_chirp": self.num_samples_per_chirp,
                "num_rx_channels": self.num_rx_channels,
                "data_type": "uint12"
            }
            return frame, metadata
            
        except queue.Empty:
            return None, None
        except Exception as e:
            logger.error(f"Error getting frame: {e}")
            return None, None

    def close(self) -> None:
        """Close the radar connection and release resources."""
        self.stop()
        self._cleanup_gpio()
        logger.debug("Radar closed")

    def get_start_frequency(self, sequence_index: int = 0) -> float:
        """
        Get the start frequency of the radar directly from registers.
        
        Args:
            sequence_index: Index of the sequence to get the start frequency for (0-3)
            
        Returns:
            Start frequency in Hz, or 0.0 if an error occurs
        """
        pll = self.pll_registers(sequence_index)
        # The frequency is calculated as: f_RF = 8 * f_SYSCLK * [4(N_DIVST + 2) + 8 + N_FSU/2^20]
        # where f_SYSCLK is typically 80 MHz and N_DIVST is the DIVSET value from PACR2
        fsu1_value = pll[0].FSU1.value
        divset_value = self.registermap.PACR2.DIVSET.value
                    
        # Convert FSU1 from 2's complement if it's negative
        if fsu1_value & 0x800000:  # Check if bit 23 is set (negative number)
            fsu1_value = fsu1_value - (1 << 24)  # Convert from 2's complement
        
        # Calculate the RF frequency using the formula from the datasheet
        frequency = 8 * self.f_sysclk * (4 * (divset_value + 2) + 8 + fsu1_value / (2**20))
        
        logger.debug(f"Start frequency for sequence {sequence_index}: {frequency/1e9:.3f} GHz "
                        f"(FSU1: {fsu1_value}, DIVSET: {divset_value})")        
        return frequency


    def pll_registers(self, sequence_index: int = 0) -> list:
        """
        Get the PLL registers for a specific sequence.
        """
        if not 0 <= sequence_index <= 3:
            raise ValueError(f"Invalid sequence index: {sequence_index}. Must be 0-3.")

        pll_registers = {
            0: [self.registermap.PLL1_0, self.registermap.PLL1_1, self.registermap.PLL1_2, self.registermap.PLL1_3,
                self.registermap.PLL1_4, self.registermap.PLL1_5, self.registermap.PLL1_6, self.registermap.PLL1_7],
            1: [self.registermap.PLL2_0, self.registermap.PLL2_1, self.registermap.PLL2_2, self.registermap.PLL2_3,
                self.registermap.PLL2_4, self.registermap.PLL2_5, self.registermap.PLL2_6, self.registermap.PLL2_7],
            2: [self.registermap.PLL3_0, self.registermap.PLL3_1, self.registermap.PLL3_2, self.registermap.PLL3_3,
                self.registermap.PLL3_4, self.registermap.PLL3_5, self.registermap.PLL3_6, self.registermap.PLL3_7],
            3: [self.registermap.PLL4_0, self.registermap.PLL4_1, self.registermap.PLL4_2, self.registermap.PLL4_3,
                self.registermap.PLL4_4, self.registermap.PLL4_5, self.registermap.PLL4_6, self.registermap.PLL4_7],
        }
        return pll_registers[sequence_index]

    def get_frequency_ramp_info(self, sequence_index: int = 0) -> dict:
        """
        Get complete frequency ramp information for a sequence.
        
        Args:
            sequence_index: Index of the sequence to get the ramp info for (0-3)
            
        Returns:
            Dictionary containing:
            - start_frequency: Start frequency in Hz
            - end_frequency: End frequency in Hz
            - ramp_slope: Frequency change per clock cycle in Hz
            - ramp_time: Total ramp time in seconds
            - ramp_bandwidth: Total frequency bandwidth in Hz
        """
        pll = self.pll_registers(sequence_index)
            
        # Read register values
        fsu1_value = pll[0].FSU1.value
        rsu1_value = pll[1].RSU1.value
        rtu1_value = pll[2].RTU1.value
            
        # Read the DIVSET value from PACR2 register
        divset_value = self.registermap.PACR2.DIVSET.value
        
        # Convert values from 2's complement if negative
        if fsu1_value & 0x800000:
            fsu1_value = fsu1_value - (1 << 24)
        if rsu1_value & 0x800000:
            rsu1_value = rsu1_value - (1 << 24)
        
        # Calculate start frequency
        start_frequency = 8 * self.f_sysclk * (4 * (divset_value + 2) + 8 + fsu1_value / (2**20))
        
        # Calculate ramp slope (frequency change per clock cycle)
        ramp_slope = 8 * self.f_sysclk * rsu1_value / (2**20)
        
        # Calculate ramp time (RTU defines number of 8-clock-cycle steps)
        ramp_time = rtu1_value * 8 * self.t_sysclk_ns * 1e-9
        
        # Calculate end frequency
        end_frequency = start_frequency + (ramp_slope * rtu1_value * 8)
        
        # Calculate total bandwidth
        ramp_bandwidth = abs(end_frequency - start_frequency)
        
        ramp_info = {
            'start_frequency': start_frequency,
            'end_frequency': end_frequency,
            'ramp_slope': ramp_slope,
            'ramp_time': ramp_time,
            'ramp_bandwidth': ramp_bandwidth
        }
        
        logger.debug(f"Frequency ramp info for sequence {sequence_index}: "
                        f"Start: {start_frequency/1e9:.3f} GHz, "
                        f"End: {end_frequency/1e9:.3f} GHz, "
                        f"Bandwidth: {ramp_bandwidth/1e9:.3f} GHz, "
                        f"Time: {ramp_time*1e6:.1f} µs")
        
        return ramp_info
        
    def get_start_frequency(self, sequence_index: int = 0) -> float:
        """
        Get the start frequency of the radar for a specific sequence.
        """
        if not 0 <= sequence_index <= 3:
            raise ValueError(f"Invalid sequence index: {sequence_index}. Must be 0-3.")
            
        info = self.get_frequency_ramp_info(sequence_index)
        return info['start_frequency']

    def get_stop_frequency(self, sequence_index: int = 0) -> float:
        """
        Get the stop frequency of the radar for a specific sequence.
        """
        if not 0 <= sequence_index <= 3:
            raise ValueError(f"Invalid sequence index: {sequence_index}. Must be 0-3.")
        
        info = self.get_frequency_ramp_info(sequence_index)
        return info['end_frequency']

    def get_ramp_slope(self, sequence_index: int = 0) -> float:
        """
        Get the ramp slope of the radar for a specific sequence.
        """
        if not 0 <= sequence_index <= 3:
            raise ValueError(f"Invalid sequence index: {sequence_index}. Must be 0-3.")
        
        info = self.get_frequency_ramp_info(sequence_index)
        return info['ramp_slope']

    def get_ramp_time(self, sequence_index: int = 0) -> float:
        """
        Get the ramp time of the radar for a specific sequence.
        """
        if not 0 <= sequence_index <= 3:
            raise ValueError(f"Invalid sequence index: {sequence_index}. Must be 0-3.")

        info = self.get_frequency_ramp_info(sequence_index)
        return info['ramp_time']

    def get_ramp_bandwidth(self, sequence_index: int = 0) -> float:
        """
        Get the ramp bandwidth of the radar for a specific sequence.
        """
        if not 0 <= sequence_index <= 3:
            raise ValueError(f"Invalid sequence index: {sequence_index}. Must be 0-3.")

        info = self.get_frequency_ramp_info(sequence_index)
        return info['ramp_bandwidth'] 

    def k_chirp(self, sequence_index: int = 0) -> float:
        """
        Get the chirp slope k (Hz/s) for the actual used chirp.

        This uses the programmed linear ramp parameters read from the device
        registers. The instantaneous slope during the up-chirp is constant and
        equals:

            k = (frequency change per SYSCLK cycle) * f_sysclk

        Returns:
            Chirp slope in Hz/s
        """
        if not 0 <= sequence_index <= 3:
            raise ValueError(f"Invalid sequence index: {sequence_index}. Must be 0-3.")

        # 'ramp_slope' from get_frequency_ramp_info is in Hz per SYSCLK cycle
        ramp_slope_per_cycle = self.get_ramp_slope(sequence_index)
        return ramp_slope_per_cycle * self.f_sysclk

    def set_start_frequency(self, frequency: float, sequence_index: int = 0) -> None:
        """
        Set the start frequency of the radar for a specific sequence.
        
        Args:
            frequency: Start frequency in Hz
            sequence_index: Index of the sequence to set the start frequency for (0-3)
        """
        pll = self.pll_registers(sequence_index)

        divset_value = self.registermap.PACR2.DIVSET.value
           
        # Calculate the FSU1 value using the inverse of the frequency formula
        # From datasheet: f_RF = 8 * f_SYSCLK * [4(N_DIVST + 2) + 8 + N_FSU/2^20]
        # Solving for N_FSU: N_FSU = 2^20 * [f_RF/(8*f_SYSCLK) - 4(N_DIVST + 2) - 8]
        
        n_fsu = int((2**20) * (frequency / (8 * self.f_sysclk) - 4 * (divset_value + 2) - 8))
        
        # Check if the calculated value is within the valid range [-2^23, 2^23-1]
        if n_fsu < -(2**23) or n_fsu > (2**23 - 1):
            raise ValueError(f"Calculated FSU1 value {n_fsu} is out of valid range [-8388608, 8388607]")
        
        # Convert to 2's complement if negative
        if n_fsu < 0:
            n_fsu = n_fsu + (1 << 24)
        
        # Set the FSU1 value in the register
        pll[0].FSU1.value = n_fsu
        
        logger.debug(f"Start frequency for sequence {sequence_index} set to {frequency/1e9:.3f} GHz "
                    f"(FSU1: {n_fsu}, DIVSET: {divset_value})")

    def set_stop_frequency(self, frequency: float, sequence_index: int = 0) -> None:
        """
        Set the stop frequency of the radar for a specific sequence.
        
        Args:
            frequency: Stop frequency in Hz
            sequence_index: Index of the sequence to set the stop frequency for (0-3)
        """
        pll = self.pll_registers(sequence_index)
        # end_frequency = start_frequency + (ramp_slope * rtu1_value * 8)
        # so we need to adjust the ramp slope to get the correct stop frequency
        rtu1_value = pll[2].RTU1.value
        start_frequency = self.get_start_frequency(sequence_index)
        new_ramp_slope = (frequency - start_frequency) / (8 * rtu1_value)

        rsu1_value = new_ramp_slope * (2**20) / (8 * self.f_sysclk)
        if rsu1_value < -(2**23) or rsu1_value > (2**23 - 1):
            raise ValueError(f"Calculated RSU1 value {rsu1_value} is out of valid range [-8388608, 8388607]")
        
        # Convert to 2's complement if negative
        rsu1_value = int(rsu1_value)
        if rsu1_value < 0:
            rsu1_value = rsu1_value + (1 << 24)
        pll[1].RSU1.value = rsu1_value
        logger.debug(f"Stop frequency for sequence {sequence_index} set to {frequency/1e9:.3f} GHz ")

    @property
    def frame_length(self) -> float:
        """
        Get the number of shape groups in one frame
        """
        if self._frame_length_cache is None:
            self._frame_length_cache = self.registermap.CCR2.FRAME_LEN.value + 1
        return self._frame_length_cache
    
    @frame_length.setter
    def frame_length(self, length: float) -> None:
        """
        Set the number of repetitions of the chirp for one frame
        """
        self.registermap.CCR2.FRAME_LEN.value = length - 1
        self._frame_length_cache = self.registermap.CCR2.FRAME_LEN.value + 1

    def get_chirp_duration(self, sequence_index: int = 0) -> float:
        """
        Get the duration of the chirp in seconds
        """
        pll = self.pll_registers(sequence_index)
        rtu1_value = pll[2].RTU1.value
        # RTU defines the number of clock cycles for the Upchirp
        chirp_duration_ns = rtu1_value * 8 * self.t_sysclk_ns
        chirp_duration = chirp_duration_ns *1e-9
        return chirp_duration

    def set_chirp_duration(self, duration: float, sequence_index: int = 0) -> None:
        """
        Set the duration of the chirp in seconds
        """
        duration_ns = duration * 1e9
        pll = self.pll_registers(sequence_index)
        pll[2].RTU1.value = int(duration_ns / 8 / self.t_sysclk_ns)

    def get_chirp_repetition_time(self, sequence_index: int = 0) -> float:
        """
        Get the repetition time of the chirp in seconds
        """
        return self.shape_group_duration(sequence_index)

    @property
    def frame_duration(self) -> float:
        """
        Get the total frame duration in seconds.
        
        This calculates the complete frame duration including:
        - Frame execution time (all shape groups)
        - Inter-frame delay (T_FED)
        
        Returns:
            Frame duration in seconds
        """
        frame_execution_time = self.frame_length * self.shape_group_duration
        return frame_execution_time + self.inter_frame_delay
    
    @frame_duration.setter
    def frame_duration(self, period: float) -> None:
        """Set the frame repetition period in seconds"""
        # you can only set the inter-frame delay, the frame duration is set by the modulation parameters
        frame_execution_time = self.frame_length * self.shape_group_duration
        inter_frame_delay = period - frame_execution_time
        # Calculate TR_FED and TR_FED_MUL values for the desired inter-frame delay
        self.inter_frame_delay = inter_frame_delay
        
        logger.debug(f"Frame repetition period set to {period*1e3:.6f} ms "
                    f"(inter_frame_delay={inter_frame_delay*1e3:.6f} ms)")

    @property
    def shape_group_duration(self, sequence_index: int = 0) -> float:
        """
        Calculate the duration of a single shape group in seconds.
        
        This includes:
        - Chirp duration (T_RAMP)
        - End of chirp delays (T_EDU/T_EDD)
        - Shape end delay (T_SED)
        - Interchirp delays
        
        Returns:
            Shape group duration in seconds
        """
        pll = self.pll_registers(sequence_index)
        # Get chirp duration (T_RAMP)
        rtu1 = pll[2].RTU1.value
        chirp_duration = rtu1 * 8 * self.t_sysclk_ns
        
        # Get end of chirp delays
        tr_edu1 = pll[2].TR_EDU1.value
        if tr_edu1 == 0:
            end_chirp_delay = 2 * self.t_sysclk_ns 
        else:
            end_chirp_delay = (8 * tr_edu1 + 5) * self.t_sysclk_ns
        
        # Get shape end delay (T_SED)
        tr_sed = pll[7].TR_SED.value
        tr_sed_mul = pll[7].TR_SED_MUL.value
        if tr_sed == 0:
            shape_end_delay = self.t_sysclk_ns
        else:
            shape_end_delay = (tr_sed * (2 ** tr_sed_mul) * 8 + tr_sed_mul + 3) * self.t_sysclk_ns
        
        # Get shape repetitions
        reps = pll[7].REPS.value
        shape_repetitions = 2 ** reps
        
        # Calculate total shape group duration
        single_shape_duration = chirp_duration + end_chirp_delay
        shape_group_duration = shape_repetitions * single_shape_duration + shape_end_delay
        
        return shape_group_duration*1e-9
    
    @property
    def inter_frame_delay(self) -> float:
        """
        Calculate the inter-frame delay (T_FED) in nanoseconds (idle time between frames).
        
        Returns:
            Inter-frame delay in seconds
        """
        tr_fed = self.registermap.CCR1.TR_FED.value
        tr_fed_mul = self.registermap.CCR1.TR_FED_MUL.value
        
        if tr_fed == 0:
            return self.t_sysclk_ns * 1e-9
        else:
            return (tr_fed * (2 ** tr_fed_mul) * 8 + tr_fed_mul + 3) * self.t_sysclk_ns * 1e-9
    
    @inter_frame_delay.setter
    def inter_frame_delay(self, delay_seconds: float) -> None:
        """
        Set the inter-frame delay by calculating TR_FED and TR_FED_MUL values.
        
        Args:
            delay_seconds: Desired inter-frame delay in seconds
        """
        # Convert to clock cycles: delay_ns = (TR_FED * 2^TR_FED_MUL * 8 + TR_FED_MUL + 3) * t_sysclk_ns
        delay_cycles = delay_seconds * 1e9 / self.t_sysclk_ns
        
        # Minimum delay is 1 clock cycle (TR_FED=0)
        if delay_cycles <= 1:
            self.registermap.CCR1.TR_FED.value = 0
            self.registermap.CCR1.TR_FED_MUL.value = 0
            return
        
        # Find best TR_FED_MUL (0-10) that keeps TR_FED <= 255
        for tr_fed_mul in range(11):
            # TR_FED = (delay_cycles - TR_FED_MUL - 3) / (2^TR_FED_MUL * 8)
            tr_fed = (delay_cycles - tr_fed_mul - 3) / ((2 ** tr_fed_mul) * 8)
            
            if tr_fed <= 255:
                self.registermap.CCR1.TR_FED.value = int(tr_fed)
                self.registermap.CCR1.TR_FED_MUL.value = tr_fed_mul
                return
        
        # If we get here, use maximum values
        self.registermap.CCR1.TR_FED.value = 255
        self.registermap.CCR1.TR_FED_MUL.value = 10

    @property
    def num_samples_per_chirp(self) -> int:
        """
        Get the number of samples per chirp.
        """
        return self.registermap.PLL1_3.APU1.value
    
    @num_samples_per_chirp.setter
    def num_samples_per_chirp(self, value: int) -> None:
        """
        Set the number of samples per chirp.
        """
        self.registermap.PLL1_3.APU1.value = value

    @property
    def adc_sample_rate(self) -> float:
        """
        Get the ADC sample rate in Hz.
        """
        sample_rate = self.f_sysclk / self.registermap.ADC0.ADC_DIV.value
        return sample_rate
    
    @adc_sample_rate.setter
    def adc_sample_rate(self, value: float) -> None:
        """
        Set the ADC sample rate in Hz.
        """
        adc_div_value = int(self.f_sysclk / value)
        self.registermap.ADC0.ADC_DIV.value = adc_div_value
        logger.debug(f"ADC sample rate set to {value/1e6:.2f} MHz")


    @property
    def output_power(self) -> int:
        """
        Get the output power level for the upchirp.
        """
        return self.registermap.CSU_1.TX_DAC.value
    
    @output_power.setter
    def output_power(self, value: int) -> None:
        """
        Set the output power level for the upchirp.
        """
        self.registermap.CSU_1.TX_DAC.value = value
        logger.debug(f"Output power set to {value}")
