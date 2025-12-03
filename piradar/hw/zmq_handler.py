"""
ZeroMQ Handler for PiRadar Data Streaming
Provides high-performance one-way data streaming for radar frames and status.
Commands are now handled via HTTP API.
"""

import asyncio
import zmq.asyncio
import pickle
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class MessageType(Enum):
    """Types of messages that can be sent/received"""
    RADAR_FRAME = "radar_frame"
    STATUS = "status"
    # Legacy types maintained for compatibility in enums, though not used in streaming
    COMMAND = "command"
    RESPONSE = "response"
    ERROR = "error"
    CONFIG = "config"


@dataclass
class RadarFrame:
    """Radar frame data structure"""
    frame_id: int
    timestamp: float
    data: np.ndarray
    timestamp_ns: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Status:
    """Status message structure"""
    timestamp: float
    radar_status: str
    frame_count: int
    fps: float
    uptime: float
    memory_usage: Optional[float] = None


class RadarStreamer:
    """
    Async ZeroMQ handler for radar data streaming (PUB/SUB)
    
    Supports:
    - Publishing radar frames (PUB socket)
    - Publishing status updates (PUB socket)
    
    Note: Command handling has been moved to the HTTP API.
    """
    
    def __init__(self, 
                 data_port: int = 5555,
                 status_port: int = 5557,
                 host: str = "*",
                 context: Optional[zmq.asyncio.Context] = None):
        """
        Initialize radar streamer
        
        Args:
            data_port: Port for publishing radar data
            status_port: Port for publishing status updates
            host: Host to bind to (default: all interfaces)
            context: Optional ZMQ context (creates new if None)
        """
        self.data_port = data_port
        self.status_port = status_port
        self.host = host
        
        # ZMQ setup
        self.context = context or zmq.asyncio.Context()
        self.data_socket = None
        self.status_socket = None
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = 0
        self.current_fps = 0.0  # Track instantaneous FPS
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize sockets
        self._setup_sockets()
    
    def _setup_sockets(self):
        """Set up ZMQ sockets"""
        try:
            self._create_data_socket()
            self._create_status_socket()
        except (zmq.ZMQError, OSError) as e:
            self.logger.error(f"Failed to set up ZMQ sockets: {e}")
            raise

    def _create_data_socket(self):
        """Create data publisher socket"""
        self.data_socket = self.context.socket(zmq.PUB)
        self.data_socket.setsockopt(zmq.LINGER, 0)
        self.data_socket.bind(f"tcp://{self.host}:{self.data_port}")
        self.logger.debug(f"Data publisher bound to tcp://{self.host}:{self.data_port}")

    def _create_status_socket(self):
        """Create status publisher socket"""
        self.status_socket = self.context.socket(zmq.PUB)
        self.status_socket.setsockopt(zmq.LINGER, 0)
        self.status_socket.bind(f"tcp://{self.host}:{self.status_port}")
        self.logger.debug(f"Status publisher bound to tcp://{self.host}:{self.status_port}")
    
    async def publish_frame(self, frame_data: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """
        Publish a radar frame asynchronously
        
        Args:
            frame_data: Radar frame data as numpy array
            metadata: Optional metadata to include
        """
        if self.data_socket is None:
            raise RuntimeError("Data socket not initialized")
        
        try:
            # Capture timestamps once to maintain alignment between float and ns fields
            timestamp_ns = time.time_ns()
            current_time = timestamp_ns / 1e9

            # Update statistics
            self.frame_count += 1
            
            # Calculate instantaneous FPS
            if self.last_frame_time > 0:
                fps = 1.0 / (current_time - self.last_frame_time)
            else:
                fps = 0.0
            
            self.current_fps = fps  # Store for status reporting
            self.last_frame_time = current_time
            
            # Create simple message format
            message = {
                "frame_id": self.frame_count,
                "timestamp": current_time,
                "timestamp_ns": timestamp_ns,
                "frequency": 58.0,  # Default frequency
                "data": frame_data
            }
            
            if metadata:
                message.update(metadata)
            
            serialized = pickle.dumps(message)
            await self.data_socket.send(serialized)
            
            self.logger.debug(f"Published frame {self.frame_count}: shape {frame_data.shape}")
            
        except (pickle.PickleError, zmq.ZMQError) as e:
            self.logger.error(f"Error publishing frame: {e}")
            raise
    
    async def publish_status(self, radar_status: str, memory_usage: Optional[float] = None):
        """
        Publish status update asynchronously
        
        Args:
            radar_status: Current radar status string
            memory_usage: Optional memory usage in MB
        """
        if self.status_socket is None:
            raise RuntimeError("Status socket not initialized")
        
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Use the instantaneous FPS calculated during frame publishing
            fps = self.current_fps
            
            status = Status(
                timestamp=current_time,
                radar_status=radar_status,
                frame_count=self.frame_count,
                fps=fps,
                uptime=uptime,
                memory_usage=memory_usage
            )
            
            message = {
                "type": MessageType.STATUS.value,
                "data": asdict(status)
            }
            
            serialized = pickle.dumps(message)
            await self.status_socket.send(serialized)
            
            self.logger.debug(f"Published status: {radar_status}, FPS: {fps:.1f}")
            
        except (pickle.PickleError, zmq.ZMQError) as e:
            self.logger.error(f"Error publishing status: {e}")
            raise
    
    async def close(self):
        """Close all sockets and clean up asynchronously"""
        if self.data_socket:
            self.data_socket.close()
        if self.status_socket:
            self.status_socket.close()
        
        self.logger.debug("Async radar streamer closed")


# Backward compatibility
AsyncZMQHandler = RadarStreamer


# Client classes - these still use ZMQ for data subscription
class AsyncZMQClient:
    """
    Async ZMQ client for receiving radar data
    
    Supports:
    - Subscribing to radar frames
    - Subscribing to status updates
    
    Note: Command sending should be done via HTTP client.
    """
    
    def __init__(self, 
                 data_port: int = 5555,
                 status_port: int = 5557,
                 host: str = "localhost",
                 context: Optional[zmq.asyncio.Context] = None,
                 # Deprecated args kept for compatibility but ignored or handled
                 command_port: int = 5556): 
        """
        Initialize async ZMQ client
        """
        self.data_port = data_port
        self.status_port = status_port
        self.host = host
        
        # ZMQ setup
        self.context = context or zmq.asyncio.Context()
        self.data_socket = None
        self.status_socket = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize sockets
        self._setup_sockets()
    
    def _setup_sockets(self):
        """Set up ZMQ client sockets"""
        try:
            # Data subscriber socket
            self.data_socket = self.context.socket(zmq.SUB)
            self.data_socket.connect(f"tcp://{self.host}:{self.data_port}")
            self.data_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.logger.debug(f"Data subscriber connected to tcp://{self.host}:{self.data_port}")
            
            # Status subscriber socket
            self.status_socket = self.context.socket(zmq.SUB)
            self.status_socket.connect(f"tcp://{self.host}:{self.status_port}")
            self.status_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.logger.debug(f"Status subscriber connected to tcp://{self.host}:{self.status_port}")
            
        except (zmq.ZMQError, OSError) as e:
            self.logger.error(f"Failed to set up ZMQ client sockets: {e}")
            raise
    
    async def receive_frame(self, timeout: Optional[float] = None) -> Optional[RadarFrame]:
        """
        Receive a radar frame asynchronously
        """
        if self.data_socket is None:
            raise RuntimeError("Data socket not initialized")
        
        try:
            # Async receive with timeout
            if timeout is not None:
                message = await asyncio.wait_for(
                    self.data_socket.recv(), 
                    timeout=timeout
                )
            else:
                message = await self.data_socket.recv()
            
            # Parse message
            data = pickle.loads(message)
            
            # Handle both raw dict format and MessageType format if needed
            # The streamer sends simple dict for frames now
            if isinstance(data, dict):
                if "type" in data and data["type"] == MessageType.RADAR_FRAME.value:
                    frame_data = data["data"]
                    return RadarFrame(**frame_data)
                elif "frame_id" in data and "data" in data:
                    # Simple format
                    return RadarFrame(**data)
            
            self.logger.warning(f"Received unknown message format")
            return None
                
        except asyncio.TimeoutError:
            return None
        except (pickle.PickleError, zmq.ZMQError, KeyError) as e:
            self.logger.error(f"Error receiving frame: {e}")
            return None
    
    async def receive_status(self, timeout: Optional[float] = None) -> Optional[Status]:
        """
        Receive a status update asynchronously
        """
        if self.status_socket is None:
            raise RuntimeError("Status socket not initialized")
        
        try:
            # Async receive with timeout
            if timeout is not None:
                message = await asyncio.wait_for(
                    self.status_socket.recv(), 
                    timeout=timeout
                )
            else:
                message = await self.status_socket.recv()
            
            # Parse message
            data = pickle.loads(message)
            
            if data["type"] == MessageType.STATUS.value:
                status_data = data["data"]
                return Status(**status_data)
            else:
                self.logger.warning(f"Received non-status message: {data['type']}")
                return None
                
        except asyncio.TimeoutError:
            return None
        except (pickle.PickleError, zmq.ZMQError, KeyError) as e:
            self.logger.error(f"Error receiving status: {e}")
            return None
    
    async def send_command(self, *args, **kwargs):
        """Deprecated: Use HTTP client for commands"""
        self.logger.error("ZMQ command sending is deprecated. Use HttpClient.")
        raise NotImplementedError("ZMQ command sending is deprecated. Use HTTP.")
    
    async def close(self):
        """Close all sockets and clean up asynchronously"""
        if self.data_socket:
            self.data_socket.close()
        if self.status_socket:
            self.status_socket.close()
        
        self.logger.debug("Async ZMQ client closed")


# Backward compatibility
ZMQHandler = RadarStreamer
ZMQClient = AsyncZMQClient
