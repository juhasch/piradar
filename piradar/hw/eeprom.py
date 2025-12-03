import smbus2
import time

def check_hat():
    """Check if Dreamhat is installed by checking the EEPROM"""
    bus_number = 0
    address = 0x51

    bus = smbus2.SMBus(bus_number)

    try:
        bus.read_byte(address)
    except (OSError, IOError) as e:
        raise RuntimeError("Hat not found") from e

    # Send memory address (2 bytes for 24C32)
    start_addr = 32+10
    addr_msb = (start_addr >> 8) & 0xFF
    addr_lsb = start_addr & 0xFF
    bus.write_i2c_block_data(address, addr_msb, [addr_lsb])

    num_bytes = 8
    data = []
    for i in range(num_bytes):
        try:
            byte = bus.read_byte(address)
            data.append(byte)
        except OSError as e:
            print(f"Error reading byte {i}: {e}")
            break
    # Check for "Dream RF"
    if data != [68, 114, 101, 97, 109, 32, 82, 70]:
        raise Exception("Hat not found")
    bus.close()
