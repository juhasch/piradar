import time
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel

from .bgt60tr13c import BGT60TR13C
from .radar_controller import AdaptiveRadarController

# Pydantic models for request/response
class CommandRequest(BaseModel):
    command_id: str
    parameters: Dict[str, Any] = {}
    timestamp: float = 0.0
    source: str = "client"

class CommandResponse(BaseModel):
    command_id: str
    timestamp: float
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None

class StatusResponse(BaseModel):
    timestamp: float
    radar_status: str
    frame_count: int
    fps: float
    uptime: float
    memory_usage: Optional[float] = None

class ParameterWriteRequest(BaseModel):
    parameter: str
    value: int
    source: str = "http"

class BatchWriteRequest(BaseModel):
    changes: Dict[str, int]
    source: str = "http_batch"

# Global state holder (will be populated by the server runner)
class RadarState:
    def __init__(self):
        self.radar: Optional[BGT60TR13C] = None
        self.controller: Optional[AdaptiveRadarController] = None
        self.start_time = time.time()
        self.last_command_time = time.time()
        self.frame_count = 0
        self.current_fps = 0.0

state = RadarState()
logger = logging.getLogger("piradar.api")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.debug("API Server starting up")
    yield
    # Shutdown
    logger.debug("API Server shutting down")

app = FastAPI(title="PiRadar API", lifespan=lifespan)

# Dependency to check radar availability
def get_radar():
    if state.radar is None:
        raise HTTPException(status_code=503, detail="Radar not initialized")
    return state.radar

def get_controller():
    if state.controller is None:
        raise HTTPException(status_code=503, detail="Adaptive controller not initialized")
    return state.controller

@app.post("/command", response_model=CommandResponse)
async def handle_command(cmd: CommandRequest):
    """Handle generic commands (compatibility layer)"""
    state.last_command_time = time.time()
    
    try:
        result = None
        success = True
        error_msg = None

        if cmd.command_id == "start":
            if state.radar:
                state.radar.start()
                result = {"status": "started", "message": "Radar started successfully"}
            else:
                raise HTTPException(status_code=503, detail="Radar not available")
                
        elif cmd.command_id == "stop":
            if state.radar:
                state.radar.stop()
                result = {"status": "stopped", "message": "Radar stopped successfully"}
            else:
                raise HTTPException(status_code=503, detail="Radar not available")
                
        elif cmd.command_id == "get_status":
            # Forward to status endpoint logic
            radar_config = {}
            if state.radar:
                try:
                    # Build radar config from radar object properties
                    radar_config = {
                        "start_frequency_Hz": state.radar.get_start_frequency(),
                        "stop_frequency_Hz": state.radar.get_stop_frequency(),
                        "chirp_duration_s": state.radar.get_chirp_duration(),
                        "k_chirp": state.radar.k_chirp(),
                        "frame_length": state.radar.frame_length,
                        "frame_duration_s": state.radar.frame_duration,
                        "output_power": state.radar.output_power,
                        "adc_sample_rate_Hz": state.radar.adc_sample_rate,
                    }
                except Exception as e:
                    logger.warning(f"Failed to build radar config: {e}")
                    radar_config = {}
            
            return CommandResponse(
                command_id=cmd.command_id,
                timestamp=time.time(),
                success=True,
                data={
                    "radar_status": "running" if (state.radar and state.radar.is_running) else "stopped",
                    "frame_count": state.frame_count,
                    "fps": state.current_fps,
                    "uptime": time.time() - state.start_time,
                    "radar_config": radar_config
                }
            )
            
        elif cmd.command_id == "keep_alive":
            result = {"status": "alive"}
            
        # Adaptive controller commands
        elif state.controller:
            # Map legacy command IDs to controller methods
            if cmd.command_id == "read_parameter":
                name = cmd.parameters.get("parameter")
                val = state.controller.read_parameter(name)
                if val is not None:
                    result = {"parameter": name, "value": val, "success": True}
                else:
                    success = False
                    error_msg = f"Failed to read parameter {name}"
                    
            elif cmd.command_id == "write_parameter":
                name = cmd.parameters.get("parameter")
                val = cmd.parameters.get("value")
                src = cmd.parameters.get("source", "http")
                if state.controller.write_parameter(name, val, src):
                    result = {"parameter": name, "value": val, "success": True}
                else:
                    success = False
                    error_msg = f"Failed to write parameter {name}"
            
            elif cmd.command_id == "batch_write":
                changes = cmd.parameters.get("changes", {})
                src = cmd.parameters.get("source", "http_batch")
                results = state.controller.batch_write_parameters(changes, src)
                result = {"results": results, "success": True}
                
            elif cmd.command_id == "list_parameters":
                cat = cmd.parameters.get("category")
                params = state.controller.list_parameters(cat) # Enum conversion might be needed if category is string
                result = {"parameters": params, "success": True}

            elif cmd.command_id == "get_parameter_info":
                name = cmd.parameters.get("parameter")
                info = state.controller.get_parameter_info(name)
                if info:
                    result = {"parameter_info": info, "success": True}
                else:
                    success = False
                    error_msg = f"Parameter {name} not found"

            elif cmd.command_id == "get_parameter_state":
                # Get full parameter state (all parameters, metrics, change history)
                state_data = state.controller.export_parameter_state()
                result = {"state": state_data, "success": True}

            # Register map access
            elif cmd.command_id == "read_register_field":
                reg = cmd.parameters.get("register")
                fld = cmd.parameters.get("field")
                val = state.controller.read_register_field(reg, fld)
                if val is not None:
                    result = {"register": reg, "field": fld, "value": val, "success": True}
                else:
                    success = False
                    error_msg = f"Failed to read {reg}.{fld}"

            elif cmd.command_id == "write_register_field":
                reg = cmd.parameters.get("register")
                fld = cmd.parameters.get("field")
                val = cmd.parameters.get("value")
                if state.controller.write_register_field(reg, fld, val):
                     result = {"register": reg, "field": fld, "value": val, "success": True}
                else:
                     success = False
                     error_msg = f"Failed to write {reg}.{fld}"
            
            else:
                # Unknown command
                if not result and not error_msg:
                     success = False
                     error_msg = f"Unknown command: {cmd.command_id}"

        else:
             success = False
             error_msg = "Controller not initialized"

        return CommandResponse(
            command_id=cmd.command_id,
            timestamp=time.time(),
            success=success,
            data=result,
            error_message=error_msg
        )

    except Exception as e:
        logger.error(f"Error handling command {cmd.command_id}: {e}", exc_info=True)
        return CommandResponse(
            command_id=cmd.command_id,
            timestamp=time.time(),
            success=False,
            error_message=str(e)
        )

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current radar status"""
    return StatusResponse(
        timestamp=time.time(),
        radar_status="running" if (state.radar and state.radar.is_running) else "stopped",
        frame_count=state.frame_count,
        fps=state.current_fps,
        uptime=time.time() - state.start_time
    )

@app.post("/keep_alive")
async def keep_alive():
    """Reset watchdog timer"""
    state.last_command_time = time.time()
    return {"status": "alive", "timestamp": time.time()}

# Dedicated endpoints for Adaptive Control (RESTful style)

@app.get("/parameters")
async def list_parameters(category: Optional[str] = None):
    ctrl = get_controller()
    # Convert category string to Enum if needed in controller, 
    # but controller.list_parameters likely handles strings or we need to map it.
    # Checking controller implementation: list_parameters takes ParameterCategory enum.
    
    cat_enum = None
    if category:
        # Import here to avoid circular dependency issues at module level if any
        from .radar_controller import ParameterCategory
        try:
            cat_enum = ParameterCategory(category)
        except ValueError:
            pass # Or raise error
            
    params = ctrl.list_parameters(cat_enum)
    return {"parameters": params}

@app.get("/parameters/{name}")
async def get_parameter(name: str):
    ctrl = get_controller()
    info = ctrl.get_parameter_info(name)
    if not info:
        raise HTTPException(status_code=404, detail="Parameter not found")
    return info

@app.put("/parameters/{name}")
async def update_parameter(name: str, req: ParameterWriteRequest):
    ctrl = get_controller()
    if name != req.parameter:
        raise HTTPException(status_code=400, detail="Parameter name mismatch")
    
    success = ctrl.write_parameter(name, req.value, req.source)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to write parameter")
    
    return {"parameter": name, "value": req.value, "success": True}

def create_app(radar_instance: BGT60TR13C, controller_instance: AdaptiveRadarController) -> FastAPI:
    """Factory to create app with injected dependencies"""
    state.radar = radar_instance
    state.controller = controller_instance
    state.start_time = time.time()
    return app

