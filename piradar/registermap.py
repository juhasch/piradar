"""Generic YAML-driven register map utility

Example usage:
# Load a YAML map without hardware (for testing/documentation)
rm = RegisterMap("bgt60tr13c_registermap.yaml")

# With SPI hardware
from piradar.hw import SpiInterface
spi_iface = SpiInterface(spi_bus=0, spi_dev=0, spi_speed=50_000_000)
rm = RegisterMap("bgt60tr13c_registermap.yaml", spi_iface)

# Access registers and fields
rm.MAIN.LDO_MODE.value = 1
rm.ADC0.ADC_DIV.value = 40
print(rm.MAIN.LDO_MODE.value)

# Dotted field helpers
rm.set_field("MAIN.FRAME_START", 1)
print(rm.get_field("SFCTL.FIFO_CREF").value)
"""
import yaml
from typing import Dict, Any, Optional, Protocol
from dataclasses import dataclass



class RegisterInterface(Protocol):
    """Protocol for register interface implementations"""
    def read_register(self, addr: int) -> int: ...
    def write_register(self, addr: int, value: int) -> None: ...



@dataclass
class RegisterField:
    """Represents a single field within a register"""
    name: str
    msb: int
    lsb: int
    access: str
    reset: int
    desc: str
    _parent: Optional['Register'] = None
    
    @property
    def value(self) -> int:
        """Get the current value of this field from the parent register"""
        if self._parent is None:
            raise RuntimeError("Field not properly initialized")
        return self._parent._get_field_value(self)
    
    @value.setter
    def value(self, new_value: int) -> None:
        """Set the value of this field in the parent register"""
        if self._parent is None:
            raise RuntimeError("Field not properly initialized")
        self._parent._set_field_value(self, new_value)
    
    @property
    def mask(self) -> int:
        """Get the bit mask for this field"""
        return ((1 << (self.msb - self.lsb + 1)) - 1) << self.lsb
    
    @property
    def max_value(self) -> int:
        """Get the maximum value this field can hold"""
        return (1 << (self.msb - self.lsb + 1)) - 1


class Register:
    """Represents a single register with its fields"""
    
    def __init__(self, name: str, addr: int, width: int, desc: str, reset: int, 
                 fields: list, interface: Optional[RegisterInterface] = None):
        """Initialize a register with its fields.
        
        Args:
            name: Register name
            addr: Register address
            width: Register width in bits
            desc: Register description
            reset: Reset/default value
            fields: List of field definitions
            interface: Optional hardware interface for read/write operations
        """
        self.name = name
        self.addr = addr
        self.width = width
        self.desc = desc
        self.reset = reset
        self._interface = interface
        self._fields: Dict[str, RegisterField] = {}
        self._cached_value: Optional[int] = None
        
        # Create field objects
        for field_data in fields:
            field = RegisterField(**field_data)
            field._parent = self
            self._fields[field.name] = field
    
    def __getattr__(self, name: str) -> RegisterField:
        """Get a field by name"""
        if name in self._fields:
            return self._fields[name]
        raise AttributeError(f"Field '{name}' not found in register '{self.name}'")
    
    def __getitem__(self, name: str) -> RegisterField:
        """Get a field by name using bracket notation"""
        return self.__getattr__(name)
    
    @property
    def value(self) -> int:
        """Get the full register value"""
        if self._interface is None:
            return self._cached_value if self._cached_value is not None else self.reset
        try:
            self._cached_value = self._interface.read_register(self.addr)
            return self._cached_value
        except (OSError, RuntimeError) as e:
            if self._cached_value is not None:
                return self._cached_value
            raise RuntimeError(f"Failed to read register: {e}")
    
    @value.setter
    def value(self, new_value: int) -> None:
        """Set the full register value"""
        # Enforce register width and type
        if new_value < 0:
            raise ValueError("Register value must be non-negative")
        width_mask = (1 << self.width) - 1 if self.width < 64 else ((1 << 64) - 1)
        if new_value & ~width_mask:
            raise ValueError(f"Value 0x{new_value:X} exceeds register width {self.width} for {self.name}")
        new_value &= width_mask

        if self._interface is None:
            self._cached_value = new_value
            return
        try:
            self._interface.write_register(self.addr, new_value)
            self._cached_value = new_value
        except (OSError, RuntimeError) as e:
            raise RuntimeError(f"Failed to write register: {e}")
    
    def _get_field_value(self, field: RegisterField) -> int:
        """Get the value of a specific field from the register"""
        try:
            reg_value = self.value
            return (reg_value & field.mask) >> field.lsb
        except RuntimeError:
            if self._cached_value is not None:
                return (self._cached_value & field.mask) >> field.lsb
            return field.reset
    
    def _set_field_value(self, field: RegisterField, new_value: int) -> None:
        """Set the value of a specific field in the register"""
        # Enforce access permissions
        acc = (field.access or "").upper()
        if 'W' not in acc and '1C' not in acc:
            # No write capability indicated
            raise PermissionError(f"Field {self.name}.{field.name} is not writeable (access='{field.access}')")
        if new_value > field.max_value:
            raise ValueError(f"Value {new_value} exceeds maximum {field.max_value} for field {field.name}")
        
        # Read current register value for read-modify-write
        # If this fails, let the exception propagate - we shouldn't guess the register value
        reg_value = self.value
        
        # Clear the field bits and set the new field value
        reg_value &= ~field.mask
        reg_value |= (new_value << field.lsb) & field.mask
        
        # Write the modified value back
        self.value = reg_value
        
        # Note: W1C (Write-1-to-Clear) fields auto-clear in hardware after writing 1,
        # but we don't simulate this in software. The hardware handles it automatically.
    
    def reset_to_default(self) -> None:
        """Reset this register to its default value"""
        self.value = self.reset
    
    def get_fields(self) -> Dict[str, RegisterField]:
        """Get all fields in this register"""
        return self._fields.copy()
    
    def __repr__(self) -> str:
        try:
            current_value = f"0x{self.value:06X}"
        except RuntimeError:
            if self._cached_value is not None:
                current_value = f"0x{self._cached_value:06X} (cached)"
            else:
                current_value = "N/A (no interface)"
        return f"Register({self.name}, addr=0x{self.addr:02X}, value={current_value})"


class RegisterMap:
    """Main register map class that provides access to all registers"""
    
    def __init__(self, registermap_path: str = "bgt60tr13c_registermap.yaml", interface: Optional[RegisterInterface] = None): 
        """Initialize RegisterMap.
        
        Args:
            interface: Optional hardware interface for register operations
        """
        self._interface = interface
        self._registers: Dict[str, Register] = {}
        self._load_register_map(registermap_path)
        self._path = registermap_path
    
    def _load_register_map(self, registermap_path: str) -> None:
        """Load the register definitions from the YAML file
        
        Args:
            registermap_path: Path to the register map YAML file
        """
        try:
            # If it's just a filename (no path separators), try to load from package resources
            if '/' not in registermap_path and '\\' not in registermap_path:
                import importlib.resources
                resource_path = importlib.resources.files("piradar") / registermap_path
                if resource_path.exists():
                    registermap_path = str(resource_path)
            
            with open(registermap_path, 'r') as file:
                register_data = yaml.safe_load(file)
            
            if not register_data or 'registers' not in register_data:
                raise RuntimeError("Invalid register map file format: 'registers' section not found")
            
            registers = register_data['registers']
            
            for register_name, reg_info in registers.items():
                try:
                    # Parse register information
                    addr = int(reg_info['addr'], 16) if isinstance(reg_info['addr'], str) else reg_info['addr']
                    reset = int(reg_info['reset'], 16) if isinstance(reg_info['reset'], str) else reg_info['reset']
                    
                    # Convert fields from dict format to list format
                    fields_list = []
                    if 'fields' in reg_info:
                        for field_name, field_data in reg_info['fields'].items():
                            field_data['name'] = field_name
                            fields_list.append(field_data)
                    
                    register = Register(
                        name=register_name,
                        addr=addr,
                        width=reg_info['width'],
                        desc=reg_info['desc'],
                        reset=reset,
                        fields=fields_list,
                        interface=self._interface
                    )
                    self._registers[register_name] = register
                    
                except KeyError as e:
                    # Use logging instead of print for better error handling
                    import logging
                    logging.warning(f"Missing required field '{e}' in register {register_name}")
                    continue
                except ValueError as e:
                    import logging
                    logging.warning(f"Invalid value in register {register_name}: {e}")
                    continue
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"Register map file not found: {registermap_path}")
        except yaml.YAMLError as e:
            raise RuntimeError(f"Error parsing register map YAML: {e}")
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Error loading register map: {e}")
    
    def reload(self, registermap_path: Optional[str] = None) -> None:
        """Reload the register map from disk (optionally a new path)."""
        if registermap_path is None:
            registermap_path = self._path
        self._registers.clear()
        self._load_register_map(registermap_path)
        self._path = registermap_path
    
    def __getattr__(self, name: str) -> Register:
        """Get a register by name"""
        if name in self._registers:
            return self._registers[name]
        raise AttributeError(f"Register '{name}' not found")

    # Convenience helpers used by higher-level controllers
    def get_register(self, name: str) -> Optional[Register]:
        """Return a Register by name, or None if not found."""
        return self._registers.get(name)

    def list_registers(self) -> list[str]:
        """List all register names in the map."""
        return list(self._registers.keys())

    def get_register_by_addr(self, addr: int) -> Optional[Register]:
        """Find a Register by its address value, if any."""
        for reg in self._registers.values():
            if reg.addr == addr:
                return reg
        return None
    
    def iter_registers(self):
        """Iterate over (name, Register) pairs in address order."""
        return iter(sorted(self._registers.items(), key=lambda kv: kv[1].addr))
    
    def __iter__(self):
        """Iterate over register names in address order."""
        for name, _ in self.iter_registers():
            yield name
    
    def __len__(self) -> int:
        return len(self._registers)
    
    def __getitem__(self, name: str) -> Register:
        """Get a register by name using bracket notation"""
        return self.__getattr__(name)
    
    # (duplicate helpers removed)
    
    def reset_all_registers(self) -> None:
        """Reset all registers to their default values"""
        for register in self._registers.values():
            register.reset_to_default()
    
    def get_register_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a register"""
        register = self.get_register(name)
        if register is None:
            return None
        
        return {
            'name': register.name,
            'address': f"0x{register.addr:02X}",
            'width': register.width,
            'description': register.desc,
            'reset_value': f"0x{register.reset:06X}",
            'current_value': f"0x{register.value:06X}" if self._interface else "N/A",
            'fields': {name: {
                'msb': field.msb,
                'lsb': field.lsb,
                'access': field.access,
                'reset': field.reset,
                'description': field.desc,
                'current_value': field.value if self._interface else "N/A"
            } for name, field in register.get_fields().items()}
        }
    
    # Utility helpers
    def read_all(self) -> Dict[str, int]:
        """Read all register values into a dict {name: value}. In offline mode returns cached/reset values."""
        out: Dict[str, int] = {}
        for name, reg in self._registers.items():
            out[name] = reg.value
        return out
    
    def write_all(self, values: Dict[str, int]) -> None:
        """Write multiple registers from a dict {name: value}. Ignores unknown names."""
        for name, val in values.items():
            reg = self._registers.get(name)
            if reg is None:
                continue
            reg.value = val
    
    def get_field(self, dotted: str) -> RegisterField:
        """Get a field by dotted path 'REGISTER.FIELD'."""
        try:
            reg_name, field_name = dotted.split('.', 1)
        except ValueError:
            raise KeyError("Field path must be of the form 'REGISTER.FIELD'")
        reg = self.get_register(reg_name)
        if not reg:
            raise KeyError(f"Unknown register '{reg_name}'")
        return reg[field_name]
    
    def set_field(self, dotted: str, value: int) -> None:
        """Set a field by dotted path 'REGISTER.FIELD' to value."""
        fld = self.get_field(dotted)
        fld.value = value

    
