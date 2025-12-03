import logging
import threading
from typing import List

import spidev
SPIDEV_AVAILABLE = True

BGT60TRXX_SPI_WR_OP_MSK = 0x01000000
BGT60TRXX_SPI_REGADR_MSK = 0xFE000000
BGT60TRXX_SPI_REGADR_POS = 25
BGT60TRXX_SPI_DATA_MSK = 0x00FFFFFF
BGT60TRXX_SPI_DATA_POS = 0
BGT60TRXX_SPI_BURST_MODE_CMD = 0xFF000000
BGT60TRXX_SPI_BURST_MODE_SADR_POS = 17

BGT60TRXX_REG_GSR0_FOU_ERR_MSK = 0x08
BGT60TRXX_REG_GSR0_MISO_HS_READ_MSK = 0x04
BGT60TRXX_REG_GSR0_SPI_BURST_ERR_MSK = 0x02
BGT60TRXX_REG_GSR0_CLK_NUM_ERR_MSK = 0x01

BGT60TRXX_REG_MAIN = 0x00
BGT60TRXX_REG_SFCTL = 0x06
BGT60TRXX_REG_FIFO_TR13C = 0x60

BGT60TRXX_REG_SFCTL_MISO_HS_READ_MSK = 0x010000
BGT60TRXX_REG_MAIN_FRAME_START_MSK = 0x000001

BGT60TRXX_REG_FSTAT_TR13C_FIFO_SIZE = 8192


class GSRRegisterError(Exception):
    """Raised when GSR register errors are detected."""
    pass

class FIFOParameterError(Exception):
    """Raised when FIFO parameter values are invalid."""
    pass


class SpiInterface:
    """Interface for SPI communication."""
    def __init__(self, spi_bus: int, spi_dev: int, spi_speed: int):
        if not SPIDEV_AVAILABLE:
            raise ImportError(
                "spidev is required for SPI communication. "
                "Install with: pip install piradar[hardware] or pip install spidev"
            )
        
        self._spi = spidev.SpiDev()
        self._spi.open(spi_bus, spi_dev)
        self._spi.max_speed_hz = spi_speed
        self._spi.mode = 0
        self._last_gsr_reg = 0
        self._io_lock = threading.Lock()


    @property
    def max_speed_hz(self) -> int:
        """Get the maximum speed of the SPI interface."""
        return self._spi.max_speed_hz

    def write_register(self, reg_addr: int, data: int) -> int:
        """
        Set a register value via SPI.
        
        Args:
            reg_addr: Register address
            data: Data to write
            
        Returns:
            Response from the device
        """
        logging.debug(f"Writing register {hex(reg_addr)} with data {hex(data)}")
        with self._io_lock:
            # Build SPI command
            temp = ((reg_addr << BGT60TRXX_SPI_REGADR_POS) & BGT60TRXX_SPI_REGADR_MSK |
                    BGT60TRXX_SPI_WR_OP_MSK |
                    (data << BGT60TRXX_SPI_DATA_POS) & BGT60TRXX_SPI_DATA_MSK)
            
            # Convert to 4-byte transmission format
            tx_data = [(temp >> 24) & 0xFF, (temp >> 16) & 0xFF, 
                       (temp >> 8) & 0xFF, temp & 0xFF]
            
            # Send data and receive response
            rx_data = self._spi.xfer2(tx_data)
            result = (rx_data[0] << 24) | (rx_data[1] << 16) | (rx_data[2] << 8) | (rx_data[3])
            self._last_gsr_reg = rx_data[0]
            
            return result
    
    def write_register_with_speed_optimization(self, reg_addr: int, data: int) -> int:
        """
        Set a register value via SPI with automatic speed-dependent optimizations.
        
        This method automatically handles speed-dependent register modifications,
        such as setting MISO high-speed read mode for high-speed SPI operations.
        
        Args:
            reg_addr: Register address
            data: Data to write
            
        Returns:
            Response from the device
        """
        # Apply speed-dependent optimizations
        if reg_addr == BGT60TRXX_REG_SFCTL:
            if self.max_speed_hz > 20_000_000:
                data |= BGT60TRXX_REG_SFCTL_MISO_HS_READ_MSK
            else:
                data &= ~BGT60TRXX_REG_SFCTL_MISO_HS_READ_MSK
        
        # Use the standard write_register method
        return self.write_register(reg_addr, data)
    
    def read_register(self, reg_addr: int) -> int:
        """
        Get a register value via SPI.
        
        Args:
            reg_addr: Register address
            
        Returns:
            Register value
        """
        with self._io_lock:
            # Build SPI read command
            temp = (reg_addr << BGT60TRXX_SPI_REGADR_POS) & BGT60TRXX_SPI_REGADR_MSK
        
            # Convert to 4-byte transmission format
            tx_data = [(temp >> 24) & 0xFF, (temp >> 16) & 0xFF, 
                    (temp >> 8) & 0xFF, temp & 0xFF]
            
            # Send read command and receive data
            rx_data = self._spi.xfer2(tx_data)
            result = (rx_data[0] << 24) | (rx_data[1] << 16) | (rx_data[2] << 8) | rx_data[3]
            self._last_gsr_reg = rx_data[0]
            
            # Mask to 24 bits for BGT60TR13C registers (all are 24-bit)
            result &= 0xFFFFFF
            
        if reg_addr != 0x5f:  # Don't log FSTAT register
            logging.debug(f"Read register {hex(reg_addr)} with data {hex(result)}")
        return result
    def _get_fifo_data(self, num_samples: int) -> List[int]:
        """
        Read data from FIFO.
        
        Args:
            num_samples: Number of samples to read
            
        Returns:
            List of FIFO data bytes
            
        Raises:
            FIFOParameterError: If num_samples is invalid
            GSRRegisterError: If GSR register errors are detected
        """
        if not (0 < (num_samples >> 1) <= BGT60TRXX_REG_FSTAT_TR13C_FIFO_SIZE and num_samples % 2 == 0):
            logging.error(f"Invalid num_samples for FIFO read: {num_samples}")
            raise FIFOParameterError("Invalid num_samples for FIFO read")
        
        with self._io_lock:
            # Build burst mode command
            tx_data = ([BGT60TRXX_SPI_BURST_MODE_CMD >> 24 & 0xFF,
                       BGT60TRXX_REG_FIFO_TR13C << BGT60TRXX_SPI_BURST_MODE_SADR_POS >> 16 & 0xFF,
                       0x00, 0x00] + 
                      [0x00] * ((num_samples >> 1) * 3))
            
            rx_data = self._spi.xfer3(tx_data)
        self._last_gsr_reg = rx_data[0]
        idx = 1
        sync_word = (rx_data[idx] << 16) | (rx_data[idx+1] << 8) | rx_data[idx+2]
        assert sync_word == 0

        try:
            self._check_gsr_register()
        except GSRRegisterError:
            logging.error("GSR Error detected during FIFO read")
            self._print_gsr_register()
            raise
        
        return rx_data[4:]

    def _check_gsr_register(self) -> None:
        """
        Check GSR register for errors.
        
        Raises:
            GSRRegisterError: If GSR register errors are detected
        """
        error_mask = (BGT60TRXX_REG_GSR0_FOU_ERR_MSK | 
                     BGT60TRXX_REG_GSR0_SPI_BURST_ERR_MSK | 
                     BGT60TRXX_REG_GSR0_CLK_NUM_ERR_MSK)
        
        if self._last_gsr_reg & error_mask:
            raise GSRRegisterError("GSR register errors detected.")     

    def _print_gsr_register(self) -> None:
        """Print GSR register status."""
        if self._last_gsr_reg & BGT60TRXX_REG_GSR0_FOU_ERR_MSK:
            logging.info("GSR FIFO OVERFLOW/UNDERFLOW ERROR: 1")
        else:
            logging.info("GSR FIFO OVERFLOW/UNDERFLOW ERROR: 0")
        
        if self._last_gsr_reg & BGT60TRXX_REG_GSR0_MISO_HS_READ_MSK:
            logging.info("GSR MISO HS: 1")
        else:
            logging.info("GSR MISO HS: 0")
        
        if self._last_gsr_reg & BGT60TRXX_REG_GSR0_SPI_BURST_ERR_MSK:
            logging.info("GSR SPI BURST ERR: 1")
        else:
            logging.info("GSR SPI BURST ERR: 0")
        
        if self._last_gsr_reg & BGT60TRXX_REG_GSR0_CLK_NUM_ERR_MSK:
            logging.info("GSR CLOCK NUMBER ERR: 1")
        else:
            logging.info("GSR CLOCK NUMBER ERR: 0")

    def set_register_bit(self, reg_addr: int, bit_mask: int, set_bit: bool = True) -> int:
        """
        Set or clear a specific bit in a register.
        
        This method reads the current register value, modifies the specified bit,
        and writes the result back. It automatically applies speed optimizations
        if applicable.
        
        Args:
            reg_addr: Register address
            bit_mask: Bit mask for the bit to modify
            set_bit: True to set the bit, False to clear it
            
        Returns:
            Response from the device after writing
        """
        # Read current register value
        current_value = self.read_register(reg_addr)
        
        # Modify the specified bit
        if set_bit:
            new_value = current_value | bit_mask
        else:
            new_value = current_value & ~bit_mask
        
        # Write the modified value with speed optimization
        return self.write_register_with_speed_optimization(reg_addr, new_value)
    
    def start_frame(self) -> int:
        """
        Start a radar frame by setting the frame start bit in the MAIN register.
        
        This method sets the BGT60TRXX_REG_MAIN_FRAME_START_MSK bit
        in the BGT60TRXX_REG_MAIN register to initiate frame collection.
        
        Returns:
            Response from the device after writing
        """
        return self.set_register_bit(BGT60TRXX_REG_MAIN, BGT60TRXX_REG_MAIN_FRAME_START_MSK, set_bit=True)

    def stop_frame(self) -> int:
        """
        Stop a radar frame by clearing the frame start bit in the MAIN register.
        
        Returns:
            Response from the device after writing
        """
        return self.set_register_bit(BGT60TRXX_REG_MAIN, BGT60TRXX_REG_MAIN_FRAME_START_MSK, set_bit=False)
