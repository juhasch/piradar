"""
Simplified radar configuration for a single sequence.

Contains:
- start frequency
- stop frequency  
- chirp duration
- frame length = number of shape groups (only one chirp per shape group for now)
- frame duration = overall duration until next frame starts
- output power
- adc sample rate
"""
from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
import yaml


# Device limits from datasheet
MIN_RF_HZ = int(58.0e9)
MAX_RF_HZ = int(63.5e9)
MAX_RF_BW_HZ = int(5.5e9)
MAX_SAMPLE_RATE_HZ = 4_000_000
MAX_RX_MASK = 0b111  # 3 RX channels
MAX_TX_POWER_LEVEL = 31  # 5-bit DAC


class RadarConfig(BaseModel):
    """Simplified radar configuration for a single sequence."""

    start_frequency_Hz: int = Field(
        ..., 
        ge=MIN_RF_HZ, 
        le=MAX_RF_HZ, 
        description="Start frequency in Hz"
    )
    stop_frequency_Hz: int = Field(
        ..., 
        ge=MIN_RF_HZ, 
        le=MAX_RF_HZ, 
        description="Stop frequency in Hz"
    )
    
    chirp_duration_s: float = Field(
        ..., 
        gt=0, 
        description="Chirp duration in seconds"
    )
    frame_length: int = Field(
        ..., 
        gt=0, 
        description="Number of shape groups (chirps per frame)"
    )
    frame_duration_s: float = Field(
        ..., 
        gt=0, 
        description="Overall duration until next frame starts in seconds"
    )
    
    output_power: int = Field(
        ..., 
        ge=0, 
        le=MAX_TX_POWER_LEVEL, 
        description="Output power level (0..31)"
    )
    adc_sample_rate_Hz: int = Field(
        ..., 
        gt=0, 
        le=MAX_SAMPLE_RATE_HZ, 
        description="ADC sample rate in Hz"
    )

    @model_validator(mode="after")
    def validate_frequency_range(self) -> "RadarConfig":
        if self.start_frequency_Hz >= self.stop_frequency_Hz:
            raise ValueError("start_frequency_Hz must be less than stop_frequency_Hz")
        
        bandwidth = self.stop_frequency_Hz - self.start_frequency_Hz
        if bandwidth > MAX_RF_BW_HZ:
            raise ValueError(f"Frequency bandwidth ({bandwidth} Hz) exceeds maximum ({MAX_RF_BW_HZ} Hz)")
            
        return self

    @staticmethod
    def read_yaml(file_name: str) -> RadarConfig:
        """
        Read the radar configuration from a yaml file.
        
        Args:
            file_name: Path to configuration file
            
        Returns:
            RadarConfig object
            
        Raises:
            ValueError: If configuration reading fails
        """
        try:
            with open(file_name, 'r') as file:
                yaml_data = yaml.safe_load(file)
            radar_config = RadarConfig.model_validate(yaml_data)
            return radar_config
        except (yaml.YAMLError, ValueError) as e:
            raise ValueError(f"Failed to read radar configuration from yaml file: {e}")        