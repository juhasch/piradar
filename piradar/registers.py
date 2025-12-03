"""Legacy register reader for BGT60TR13C.

This module is used to read the registers from a text format created by Infineon tools.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Register:
    addr: int
    name: str
    desc: str
    default_value: int|None = None
    reserved: bool = False

def read_registers(file_name: str | None = None) -> dict[int, Register]:
    """Read the registers from a text format created by Infineon tools.
    
    Args:
        file_name:  Textfile to read
    """
    registers = {}    
    with open(file_name, 'r') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                parts = line.split()
                if len(parts) != 3:
                    print(f"Warning: Skipping malformed line {line_num}: {line}")
                    continue
                
                name, addr_str, value_str = parts
                
                # Parse address (remove 0x prefix if present)
                addr_str_clean = addr_str.replace('0x', '')
                addr = int(addr_str_clean, 16)
                
                # Parse value (remove 0x prefix if present)
                value_str_clean = value_str.replace('0x', '')
                value = int(value_str_clean, 16)
                
                # Create register with minimal description
                desc = f"Register {name} from text config"
                registers[addr] = Register(addr, name, desc, value, False)
                
            except ValueError as e:
                print(f"Warning: Skipping invalid line {line_num}: {line} - {e}")
                continue
    
    return registers
