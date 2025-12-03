"""
Radar Controller for Adaptive Operation
Provides direct access to radar parameters through register manipulation
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .bgt60tr13c import BGT60TR13C
from ..registermap import RegisterMap, RegisterField


class ParameterCategory(Enum):
    """Categories of radar parameters"""
    FREQUENCY = "frequency"
    TIMING = "timing"
    POWER = "power"
    ANTENNA = "antenna"
    SIGNAL_PROCESSING = "signal_processing"
    ADVANCED = "advanced"


@dataclass
class RadarParameter:
    """Definition of a radar parameter - wrapper around RegisterField"""
    name: str
    category: ParameterCategory
    description: str
    units: str
    scaling_factor: float = 1.0
    read_only: bool = False
    register_field: Optional['RegisterField'] = None  # Reference to the actual register field
    
    # Properties that delegate to the register field
    @property
    def register_address(self) -> int:
        """Get register address from the register field"""
        if self.register_field is None:
            return 0
        return self.register_field._parent.addr
    

    
    @property
    def default_value(self) -> int:
        """Get default value from the register field"""
        if self.register_field is None:
            return 0
        return self.register_field.reset
    
    @property
    def min_value(self) -> int:
        """Minimum value is always 0 for register fields"""
        return 0
    
    @property
    def max_value(self) -> int:
        """Get maximum value from the register field"""
        if self.register_field is None:
            return 0
        return self.register_field.max_value
    



@dataclass
class ParameterChange:
    """Record of a parameter change"""
    timestamp: float
    parameter_name: str
    old_value: int
    new_value: int
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveRadarController:
    """
    Controller for adaptive radar operation
    
    Provides direct access to radar parameters through register manipulation
    with minimal abstraction to allow AI learning of parameter effects
    """
    
    def __init__(self, radar: BGT60TR13C):
        """
        Initialize the adaptive controller
        
        Args:
            radar: BGT60TR13C radar instance
        """
        self.radar = radar
        self.register_map = RegisterMap()
        self.logger = logging.getLogger(__name__)
        
        # Parameter definitions - direct register mappings
        self.parameters = self._define_parameters()
        
        # Change history for learning
        self.change_history: List[ParameterChange] = []
        
        # Current parameter values cache
        self._parameter_cache: Dict[str, int] = {}
        
        # Performance monitoring
        self.performance_metrics = {
            'frame_count': 0,
            'parameter_changes': 0,
            'last_optimization': 0,
            'current_fps': 0,
            'signal_quality': 0.0
        }
    
    def _define_parameters(self) -> Dict[str, RadarParameter]:
        """Define available radar parameters using the actual register map"""
        params = {}
        
        # Get the register map
        regmap = getattr(self.radar, 'registermap', None)
        if regmap is None:
            regmap = self.register_map
        
        if regmap is None:
            self.logger.warning("No register map available, returning empty parameter list")
            return params
        
        # Define high-level parameter mappings from register fields to our categories
        # This creates a clean interface while using the actual hardware definitions
        parameter_mappings = {
            # Frequency parameters
            'freq_start_up1': ('PLL1_0', 'FSU1', ParameterCategory.FREQUENCY, "SDM units", 8.0),
            'freq_start_up2': ('PLL2_0', 'FSU2', ParameterCategory.FREQUENCY, "SDM units", 8.0),
            'freq_start_up3': ('PLL3_0', 'FSU3', ParameterCategory.FREQUENCY, "SDM units", 8.0),
            'freq_start_up4': ('PLL4_0', 'FSU4', ParameterCategory.FREQUENCY, "SDM units", 8.0),
            
            # Timing parameters
            'ramp_time_up1': ('PLL1_2', 'RTU1', ParameterCategory.TIMING, "clock cycles", 800.0),
            'ramp_time_up2': ('PLL2_2', 'RTU2', ParameterCategory.TIMING, "clock cycles", 800.0),
            'ramp_time_up3': ('PLL3_2', 'RTU3', ParameterCategory.TIMING, "clock cycles", 800.0),
            'ramp_time_up4': ('PLL4_2', 'RTU4', ParameterCategory.TIMING, "clock cycles", 800.0),
            
            # Chirp repetition parameters
            'shape1_repetitions': ('PLL1_7', 'REPS', ParameterCategory.TIMING, "repetitions", 1.0),
            'shape2_repetitions': ('PLL2_7', 'REPS', ParameterCategory.TIMING, "repetitions", 1.0),
            'shape3_repetitions': ('PLL3_7', 'REPS', ParameterCategory.TIMING, "repetitions", 1.0),
            'shape4_repetitions': ('PLL4_7', 'REPS', ParameterCategory.TIMING, "repetitions", 1.0),
            
            # Bandwidth parameters (ramp slopes)
            'ramp_slope_up1': ('PLL1_1', 'RSU1', ParameterCategory.FREQUENCY, "Hz/cycle", 1.0),
            'ramp_slope_up2': ('PLL2_1', 'RSU2', ParameterCategory.FREQUENCY, "Hz/cycle", 1.0),
            'ramp_slope_up3': ('PLL3_1', 'RSU3', ParameterCategory.FREQUENCY, "Hz/cycle", 1.0),
            'ramp_slope_up4': ('PLL4_1', 'RSU4', ParameterCategory.FREQUENCY, "Hz/cycle", 1.0),
            
            # Power parameters
            'tx_power': ('CSI_1', 'TX_DAC', ParameterCategory.POWER, "power level", 1.0),
            
            # Antenna parameters
            'rx1_enable': ('CSI_0', 'RX1MIX_EN', ParameterCategory.ANTENNA, "enable", 1.0),
            'rx2_enable': ('CSI_0', 'RX2MIX_EN', ParameterCategory.ANTENNA, "enable", 1.0),
            'rx3_enable': ('CSI_0', 'RX3MIX_EN', ParameterCategory.ANTENNA, "enable", 1.0),
            'tx_enable': ('CSI_0', 'TX_EN', ParameterCategory.ANTENNA, "enable", 1.0),
            
            # Signal processing parameters
            'adc_sample_rate': ('ADC0', 'ADC_DIV', ParameterCategory.SIGNAL_PROCESSING, "divider", 0.1),
            'fifo_threshold': ('SFCTL', 'FIFO_CREF', ParameterCategory.SIGNAL_PROCESSING, "samples", 1.0),
            'acquisition_points_up1': ('PLL1_3', 'APU1', ParameterCategory.SIGNAL_PROCESSING, "samples", 1.0),
            'acquisition_points_up2': ('PLL2_3', 'APU2', ParameterCategory.SIGNAL_PROCESSING, "samples", 1.0),
            
            # Advanced parameters
            'shape1_enable': ('PLL1_7', 'SH_EN', ParameterCategory.ADVANCED, "enable", 1.0),
            'shape2_enable': ('PLL2_7', 'SH_EN', ParameterCategory.ADVANCED, "enable", 1.0),
            'shape3_enable': ('PLL3_7', 'SH_EN', ParameterCategory.ADVANCED, "enable", 1.0),
            'shape4_enable': ('PLL4_7', 'SH_EN', ParameterCategory.ADVANCED, "enable", 1.0),
            
            # Frame control parameters
            'frame_length': ('CCR2', 'FRAME_LEN', ParameterCategory.TIMING, "shape groups", 1.0),
            'max_frame_count': ('CCR2', 'MAX_FRAME_CNT', ParameterCategory.TIMING, "frames", 1.0),
        }
        
        # Create RadarParameter objects from register map
        for param_name, (register_name, field_name, category, units, scaling_factor) in parameter_mappings.items():
            try:
                register = regmap.get_register(register_name)
                if register is None:
                    self.logger.warning(f"Register {register_name} not found in register map")
                    continue
                
                try:
                    field = getattr(register, field_name)
                except AttributeError:
                    self.logger.warning(f"Field {field_name} not found in register {register_name}")
                    continue
                
                # Create RadarParameter using the register field
                param = RadarParameter(
                    name=param_name,
                    category=category,
                    description=field.desc,
                    units=units,
                    scaling_factor=scaling_factor,
                    read_only=(field.access == "RO"),
                    register_field=field
                )
                
                params[param_name] = param
                
            except (KeyError, ValueError, AttributeError) as e:
                self.logger.warning(f"Failed to create parameter {param_name}: {e}")
                continue
        
        self.logger.debug(f"Created {len(params)} radar parameters from register map")
        return params
    
    def get_parameter(self, name: str) -> Optional[RadarParameter]:
        """Get parameter definition by name"""
        return self.parameters.get(name)
    
    def get_bandwidth(self, shape: int = 1) -> Optional[float]:
        """Calculate the modulation bandwidth for a given shape.
        
        Bandwidth = ramp_slope * ramp_time * 8
        
        Args:
            shape: Shape number (1-4)
            
        Returns:
            Bandwidth in Hz, or None if parameters not available
        """
        try:
            ramp_slope_param = f'ramp_slope_up{shape}'
            ramp_time_param = f'ramp_time_up{shape}'
            
            if ramp_slope_param not in self.parameters or ramp_time_param not in self.parameters:
                return None
            
            # Get the raw register values
            ramp_slope_raw = self.read_parameter(ramp_slope_param)
            ramp_time_raw = self.read_parameter(ramp_time_param)
            
            if ramp_slope_raw is None or ramp_time_raw is None:
                return None
            
            # Calculate bandwidth: bandwidth = ramp_slope * ramp_time * 8
            # The ramp_slope is in Hz per clock cycle, ramp_time is in clock cycles
            bandwidth = ramp_slope_raw * ramp_time_raw * 8
            
            return bandwidth
            
        except (AttributeError, ValueError) as e:
            self.logger.error(f"Failed to calculate bandwidth for shape {shape}: {e}")
            return None
    
    def list_parameters(self, category: Optional[ParameterCategory] = None) -> List[str]:
        """List available parameters, optionally filtered by category"""
        if category:
            return [name for name, param in self.parameters.items() 
                   if param.category == category]
        return list(self.parameters.keys())
    
    def list_all_register_parameters(self) -> Dict[str, Any]:
        """List all available parameters from the register map"""
        # Prefer the driver's hardware-bound register map if present
        regmap = getattr(self.radar, 'registermap', None)
        if regmap is None:
            regmap = self.register_map
        if regmap is None:
            return {}
        
        all_params = {}
        
        # Get all registers from the register map
        for register_name in regmap.list_registers():
            register = regmap.get_register(register_name)
            if register:
                register_params = {
                    'address': f"0x{register.addr:04X}",
                    'width': register.width,
                    'description': register.desc,
                    'reset_value': f"0x{register.reset:06X}",
                    'fields': {}
                }
                
                # Get all fields in this register
                for field_name, field in register.get_fields().items():
                    register_params['fields'][field_name] = {
                        'msb': field.msb,
                        'lsb': field.lsb,
                        'access': field.access,
                        'reset': field.reset,
                        'description': field.desc,
                        'current_value': field.value if hasattr(self.radar, 'read_register') else "N/A"
                    }
                
                all_params[register_name] = register_params
        
        return all_params
    
    def read_register_field(self, register_name: str, field_name: str) -> Optional[int]:
        """Read a specific field from a register using the register map"""
        regmap = getattr(self.radar, 'registermap', None) or self.register_map
        if regmap is None:
            return None
        
        try:
            register = regmap.get_register(register_name)
            if register and hasattr(register, field_name):
                field = getattr(register, field_name)
                return field.value
        except (AttributeError, OSError, RuntimeError) as e:
            self.logger.error(f"Failed to read register field {register_name}.{field_name}: {e}")
        
        return None
    
    def write_register_field(self, register_name: str, field_name: str, value: int, source: str = "manual") -> bool:
        """Write a specific field to a register using the register map"""
        regmap = getattr(self.radar, 'registermap', None) or self.register_map
        if regmap is None:
            return False
        
        try:
            register = regmap.get_register(register_name)
            if register and hasattr(register, field_name):
                field = getattr(register, field_name)
                old_value = field.value
                field.value = value
                
                # Record the change
                change = ParameterChange(
                    timestamp=time.time(),
                    parameter_name=f"{register_name}.{field_name}",
                    old_value=old_value,
                    new_value=value,
                    source=source,
                    metadata={
                        'register_name': register_name,
                        'field_name': field_name,
                        'register_address': register.addr,
                        'units': 'raw'
                    }
                )
                self.change_history.append(change)
                
                # Update performance metrics
                self.performance_metrics['parameter_changes'] += 1
                
                self.logger.debug(f"Register field {register_name}.{field_name} changed from {old_value} to {value} by {source}")
                return True
        except (AttributeError, OSError, RuntimeError, ValueError) as e:
            self.logger.error(f"Failed to write register field {register_name}.{field_name}: {e}")
        
        return False
    
    def read_parameter(self, name: str) -> Optional[int]:
        """
        Read current value of a parameter directly from registers
        
        Args:
            name: Parameter name
            
        Returns:
            Current parameter value or None if error
        """
        param = self.parameters.get(name)
        if not param:
            self.logger.error(f"Unknown parameter: {name}")
            return None
        
        try:
            if param.register_field is not None:
                # Use the register field directly
                param_value = param.register_field.value
            else:
                # Fallback for parameters without register fields
                param_value = param.default_value
            
            # Cache the value
            self._parameter_cache[name] = param_value
            
            return param_value
            
        except (KeyError, AttributeError, OSError, RuntimeError) as e:
            self.logger.error(f"Failed to read parameter {name}: {e}")
            return None
    
    def write_parameter(self, name: str, value: int, source: str = "manual") -> bool:
        """
        Write new value to a parameter directly to registers
        
        Args:
            name: Parameter name
            value: New value
            source: Source of the change (e.g., "ai", "manual", "optimizer")
            
        Returns:
            True if successful, False otherwise
        """
        param = self.parameters.get(name)
        if not param:
            self.logger.error(f"Unknown parameter: {name}")
            return False
        
        if param.read_only:
            self.logger.error(f"Parameter {name} is read-only")
            return False
        
        # Convert frequency parameters from MHz to SDM units
        if param.category == ParameterCategory.FREQUENCY and 'freq_start' in name:
            # Convert frequency from MHz to SDM units using the same formula as BGT60TR13C
            try:
                # Get the DIVSET value from PACR2 register
                pacr2 = self.radar.registermap.PACR2
                divset_value = pacr2.DIVSET.value
                
                # Convert MHz to Hz
                frequency_hz = value * 1e6
                
                # Calculate the FSU value using the inverse of the frequency formula
                # From datasheet: f_RF = 8 * f_SYSCLK * [4(N_DIVST + 2) + 8 + N_FSU/2^20]
                # Solving for N_FSU: N_FSU = 2^20 * [f_RF/(8*f_SYSCLK) - 4(N_DIVST + 2) - 8]
                n_fsu = int((2**20) * (frequency_hz / (8 * self.radar.f_sysclk) - 4 * (divset_value + 2) - 8))
                
                # Check if the calculated value is within the valid range [-2^23, 2^23-1]
                if n_fsu < -(2**23) or n_fsu > (2**23 - 1):
                    self.logger.error(f"Calculated FSU value {n_fsu} is out of valid range [-8388608, 8388607] for frequency {value} MHz")
                    return False
                
                # Convert to 2's complement if negative
                if n_fsu < 0:
                    n_fsu = n_fsu + (1 << 24)
                
                # Use the converted value
                register_value = n_fsu
                self.logger.debug(f"Converted frequency {value} MHz to SDM units: {register_value}")
                
            except (ValueError, ZeroDivisionError) as e:
                self.logger.error(f"Failed to convert frequency {value} MHz to SDM units: {e}")
                return False
        else:
            # For non-frequency parameters, use the value directly
            register_value = value
        
        # Validate register value range
        if register_value < param.min_value or register_value > param.max_value:
            self.logger.error(f"Register value {register_value} out of range [{param.min_value}, {param.max_value}] for {name}")
            return False
        
        try:
            if param.register_field is not None:
                # Get old value for change tracking
                old_value = param.register_field.value
                
                # Set the new value using the register field
                param.register_field.value = register_value
                
                # Verify write
                verify_value = param.register_field.value
                
                if verify_value != register_value:
                    self.logger.error(f"Parameter write verification failed for {name}")
                    return False
            else:
                # Fallback for parameters without register fields
                old_value = param.default_value
                # For parameters without register fields, we can't actually write
                self.logger.warning(f"Parameter {name} has no register field, cannot write")
                return False
            
            # Record the change
            change = ParameterChange(
                timestamp=time.time(),
                parameter_name=name,
                old_value=old_value,
                new_value=value,  # Store the original parameter value, not the register value
                source=source,
                metadata={
                    'register_address': param.register_address,
                    'units': param.units,
                    'register_value': register_value  # Also store the actual register value
                }
            )
            self.change_history.append(change)
            
            # Update cache
            self._parameter_cache[name] = value
            
            # Update performance metrics
            self.performance_metrics['parameter_changes'] += 1
            
            self.logger.debug(f"Parameter {name} changed from {old_value} to {value} by {source}")
            return True
            
        except (KeyError, AttributeError, ValueError, OSError, RuntimeError) as e:
            self.logger.error(f"Failed to write parameter {name}: {e}")
            return False
    
    def read_all_parameters(self) -> Dict[str, int]:
        """Read all parameters and return current values"""
        values = {}
        for name in self.parameters:
            value = self.read_parameter(name)
            if value is not None:
                values[name] = value
        return values
    
    def get_parameter_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive information about a parameter"""
        param = self.parameters.get(name)
        if not param:
            return None
        
        current_value = self.read_parameter(name)
        
        # Get additional info from register map if available
        register_info = None
        regmap = getattr(self.radar, 'registermap', None) or self.register_map
        if regmap is not None:
            register = regmap.get_register_by_addr(param.register_address)
            if register:
                register_info = {
                    'register_name': register.name,
                    'register_description': register.desc,
                    'register_width': register.width,
                    'register_reset': f"0x{register.reset:06X}"
                }
        
        return {
            'name': param.name,
            'category': param.category.value,
            'description': param.description,
            'register_address': f"0x{param.register_address:04X}",
            'current_value': current_value,
            'default_value': param.default_value,
            'min_value': param.min_value,
            'max_value': param.max_value,
            'units': param.units,
            'scaling_factor': param.scaling_factor,
            'read_only': param.read_only,
            'register_info': register_info
        }
    
    def get_change_history(self, 
                          parameter_name: Optional[str] = None,
                          since: Optional[float] = None,
                          source: Optional[str] = None) -> List[ParameterChange]:
        """Get parameter change history with optional filtering"""
        history = self.change_history
        
        if parameter_name:
            history = [c for c in history if c.parameter_name == parameter_name]
        
        if since:
            history = [c for c in history if c.timestamp >= since]
        
        if source:
            history = [c for c in history if c.source == source]
        
        return history
    
    def batch_write_parameters(self, 
                              changes: Dict[str, int], 
                              source: str = "batch") -> Dict[str, bool]:
        """
        Write multiple parameters in a single batch
        
        Args:
            changes: Dict of parameter_name -> new_value
            source: Source identifier for the batch
            
        Returns:
            Dict of parameter_name -> success_status
        """
        results = {}
        
        for name, value in changes.items():
            success = self.write_parameter(name, value, source)
            results[name] = success
            
            # Small delay between writes to ensure stability
            time.sleep(0.001)
        
        return results
    
    def reset_to_defaults(self, category: Optional[ParameterCategory] = None) -> Dict[str, bool]:
        """Reset parameters to default values"""
        params_to_reset = self.list_parameters(category)
        results = {}
        
        for name in params_to_reset:
            param = self.parameters[name]
            if not param.read_only:
                success = self.write_parameter(name, param.default_value, "reset")
                results[name] = success
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        # Update frame count from radar
        if hasattr(self.radar, 'frame_count'):
            self.performance_metrics['frame_count'] = self.radar.frame_count
        
        # Calculate current FPS
        if len(self.change_history) > 1:
            recent_changes = [c for c in self.change_history 
                            if time.time() - c.timestamp < 60]  # Last minute
            self.performance_metrics['current_fps'] = len(recent_changes) / 60.0
        
        return self.performance_metrics.copy()
    
    def export_parameter_state(self) -> Dict[str, Any]:
        """Export current parameter state for AI analysis"""
        return {
            'timestamp': time.time(),
            'parameters': self.read_all_parameters(),
            'performance_metrics': self.get_performance_metrics(),
            'recent_changes': [
                {
                    'parameter': c.parameter_name,
                    'old_value': c.old_value,
                    'new_value': c.new_value,
                    'timestamp': c.timestamp,
                    'source': c.source
                }
                for c in self.change_history[-100:]  # Last 100 changes
            ]
        }
    
    def import_parameter_state(self, state: Dict[str, Any]) -> bool:
        """Import parameter state from AI analysis"""
        try:
            if 'parameters' in state:
                for name, value in state['parameters'].items():
                    if name in self.parameters:
                        self.write_parameter(name, value, "ai_import")
            return True
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Failed to import parameter state: {e}")
            return False


# Convenience functions for common operations
def create_adaptive_controller(radar: BGT60TR13C) -> AdaptiveRadarController:
    """Create an adaptive radar controller"""
    return AdaptiveRadarController(radar)


def get_parameter_categories() -> List[str]:
    """Get list of available parameter categories"""
    return [cat.value for cat in ParameterCategory]
