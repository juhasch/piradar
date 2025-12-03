"""
Configuration models for radar plotting applications.

Uses Pydantic for validation and YAML file loading.
"""

from pathlib import Path
from typing import Optional, Literal, Dict, Any, List
import yaml
from pydantic import BaseModel, Field, field_validator


class ZMQConfig(BaseModel):
    """ZMQ connection configuration."""
    host: Optional[str] = Field(default=None, description="ZMQ host (auto-discover if None)")
    data_port: int = Field(default=5555, description="ZMQ data port")
    command_port: int = Field(default=5556, description="ZMQ command port")
    status_port: int = Field(default=5557, description="ZMQ status port")
    max_drain: int = Field(default=50, description="Max messages to drain per UI cycle")


class PlotConfig(BaseModel):
    """General plotting configuration."""
    window: Literal["hamming", "hanning", "blackman", "none"] = Field(
        default="hamming", description="Window function to apply"
    )
    db_min: float = Field(default=60.0, description="Minimum dynamic range in dB for display")
    db_max: float = Field(default=100.0, description="Maximum dynamic range in dB for display")
    db_range: Optional[float] = Field(default=None, description="Dynamic range in dB for log-magnitude conversion (defaults to db_max - db_min)")
    colormap: str = Field(default="viridis", description="Matplotlib colormap")
    range_bins: Optional[int] = Field(default=None, description="Number of range bins (auto if None)")
    doppler_bins: Optional[int] = Field(default=None, description="Number of Doppler bins (auto if None)")
    
    @property
    def effective_db_range(self) -> float:
        """Get the effective db_range, calculating from db_min/db_max if not set."""
        if self.db_range is not None:
            return self.db_range
        return self.db_max - self.db_min


class RangeDopplerConfig(BaseModel):
    """Configuration specific to range-Doppler plotting."""
    zmq: ZMQConfig = Field(default_factory=ZMQConfig)
    plot: PlotConfig = Field(default_factory=PlotConfig)
    angle_range: Optional[tuple[float, float]] = Field(default=None, description="Fixed angle range in radians [min, max]. None for auto-scale (-π to +π).")
    
    @field_validator('angle_range')
    @classmethod
    def validate_angle_range(cls, v):
        if v is not None:
            if len(v) != 2:
                raise ValueError('angle_range must be a tuple/list of exactly 2 values [min, max]')
            if v[0] >= v[1]:
                raise ValueError('angle_range: min must be less than max')
        return v


class VelocityWaterfallConfig(BaseModel):
    """Configuration specific to velocity waterfall plotting."""
    zmq: ZMQConfig = Field(default_factory=ZMQConfig)
    plot: PlotConfig = Field(default_factory=PlotConfig)
    history: int = Field(default=200, description="Number of rows in waterfall history")
    positive_ranges_only: bool = Field(default=False, description="Use only positive range half for velocity vector")
    suppress_static: bool = Field(default=False, description="Enable static clutter suppression")
    suppress_alpha: float = Field(default=0.995, description="EWMA alpha for static suppression")


class TimeDomainConfig(BaseModel):
    """Configuration specific to time-domain plotting."""
    zmq: ZMQConfig = Field(default_factory=ZMQConfig)
    plot: PlotConfig = Field(default_factory=PlotConfig)
    average: bool = Field(default=False, description="Average across chirps instead of showing a single chirp")
    chirp_index: int = Field(default=-1, description="Chirp index to display (-1 = last)")
    suppress_static: bool = Field(default=False, description="Enable static clutter suppression")
    suppress_alpha: float = Field(default=0.995, description="EWMA alpha for static suppression")
    ylim: Optional[tuple[float, float]] = Field(default=None, description="Fixed y-axis limits [ymin, ymax]. None for auto-scale.")
    
    @field_validator('suppress_alpha')
    @classmethod
    def validate_alpha(cls, v):
        if not 0.0 <= v < 1.0:
            raise ValueError('suppress_alpha must be in range [0.0, 1.0)')
        return v
    
    @field_validator('ylim')
    @classmethod
    def validate_ylim(cls, v):
        if v is not None:
            if len(v) != 2:
                raise ValueError('ylim must be a tuple/list of exactly 2 values [ymin, ymax]')
            if v[0] >= v[1]:
                raise ValueError('ylim: ymin must be less than ymax')
        return v


class PeakFindingConfig(BaseModel):
    """Configuration for peak finding functionality."""
    enabled: bool = Field(default=True, description="Enable peak finding and red cross display")
    min_range_cm: float = Field(default=5.0, description="Minimum range to consider for peak finding (cm)")
    threshold: float = Field(default=0.1, description="Minimum threshold for peak detection")
    ignore_range_0: bool = Field(default=True, description="Ignore peaks at range 0 due to self-coupling")


class FrequencyDomainConfig(BaseModel):
    """Configuration specific to frequency-domain plotting."""
    zmq: ZMQConfig = Field(default_factory=ZMQConfig)
    plot: PlotConfig = Field(default_factory=PlotConfig)
    average: bool = Field(default=False, description="Average across chirps instead of showing a single chirp")
    chirp_index: int = Field(default=-1, description="Chirp index to display (-1 = last)")
    show_individual_channels: bool = Field(default=True, description="Show individual receiver channel range profiles for debugging")
    peak_finding: PeakFindingConfig = Field(default_factory=PeakFindingConfig, description="Peak finding configuration")
    integration_method: Literal["coherent", "non-coherent"] = Field(default="coherent", description="Integration method: coherent or non-coherent")

class PolarPlotConfig(BaseModel):
    """Configuration specific to polar plot."""
    zmq: ZMQConfig = Field(default_factory=ZMQConfig)
    plot: PlotConfig = Field(default_factory=PlotConfig)
    average: bool = Field(default=False, description="Average across chirps instead of showing a single chirp")
    chirp_index: int = Field(default=-1, description="Chirp index to display (-1 = last)")
    peak_finding: PeakFindingConfig = Field(default_factory=PeakFindingConfig, description="Peak finding configuration")
    show_history: bool = Field(default=True, description="Show history of peaks")
    max_history: int = Field(default=50, description="Maximum number of historical peaks to display")


class RadarCalibrationConfig(BaseModel):
    """Per-radar calibration knobs used during fusion."""
    range_offset_cm: float = Field(default=0.0, description="Static range offset applied post processing")
    doppler_offset_mps: float = Field(default=0.0, description="Static Doppler/velocity offset")
    phase_offset_rad: float = Field(default=0.0, description="Optional phase offset for coherent combining")
    amplitude_db: float = Field(default=0.0, description="Amplitude gain/attenuation for balancing feeds")


class RadarInstanceConfig(BaseModel):
    """Single radar endpoint used inside a multi-radar configuration."""
    name: str = Field(..., min_length=1, description="Human readable identifier")
    zmq: ZMQConfig = Field(default_factory=ZMQConfig, description="ZMQ connection parameters for this radar")
    calibration: RadarCalibrationConfig = Field(default_factory=RadarCalibrationConfig)
    color: Optional[str] = Field(default=None, description="Preferred plot color (hex or Matplotlib name)")
    enabled: bool = Field(default=True, description="Disable to temporarily skip subscribing")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata tags")


class MultiRadarFusionConfig(BaseModel):
    """Controls how aligned frames are combined."""
    method: Literal["per_radar", "stack", "average"] = Field(
        default="per_radar",
        description="Fusion strategy: keep per radar outputs, stack along axis 0, or average across radars",
    )
    preserve_order: bool = Field(
        default=True,
        description="Maintain config order when stacking/averaging (disable to sort by timestamp)",
    )


class MultiRadarConfig(BaseModel):
    """Top-level configuration for multi-radar processing."""
    radars: List[RadarInstanceConfig] = Field(..., min_length=1, description="Radar endpoints to subscribe to")
    tolerance_ns: int = Field(
        default=5_000_000,
        ge=0,
        description="Max allowed timestamp delta (nanoseconds) when synchronizing frames",
    )
    fusion: MultiRadarFusionConfig = Field(default_factory=MultiRadarFusionConfig)
    range_doppler_defaults: RangeDopplerConfig = Field(
        default_factory=RangeDopplerConfig,
        description="Shared range-Doppler defaults reused by multi-radar sample scripts",
    )

    @property
    def enabled_radars(self) -> List[RadarInstanceConfig]:
        """Return only enabled radars."""
        return [radar for radar in self.radars if radar.enabled]


def load_config(config_path: str, config_type: type) -> BaseModel:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        data = yaml.safe_load(f)
    
    return config_type(**data)


def save_default_config(config_path: str, config: BaseModel) -> None:
    """Save default configuration to YAML file."""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, indent=2)
