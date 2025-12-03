import asyncio
import logging
import uvicorn
import socket
import time
import signal

from .bgt60tr13c import BGT60TR13C, BGT60TR13CError
from .radar_controller import create_adaptive_controller
from .zmq_handler import RadarStreamer
from .api import create_app
from .discovery import RadarAnnouncer


def uvicorn_filter_factory():
    """Factory function for uvicorn filter"""
    class UvicornFilter(logging.Filter):
        """Filter to only show 'Uvicorn running' message at INFO level"""
        def filter(self, record):
            # Allow "Uvicorn running" message at INFO level
            if record.levelno == logging.INFO and "Uvicorn running" in record.getMessage():
                return True
            # Suppress all other INFO messages from uvicorn
            if record.levelno == logging.INFO:
                return False
            # Allow all other levels (DEBUG, WARNING, ERROR, etc.)
            return True
    return UvicornFilter()

class RadarServer:
    def __init__(self, port: int = 5555, config_file: str = None, synthetic: bool = False):
        self.port = port
        self.http_port = port + 1
        self.status_port = port + 2
        self.config_file = config_file
        self.synthetic = synthetic
        
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Components
        self.radar = None
        self.controller = None
        self.streamer = None
        self.api_server = None
        self.announcer = None
        
    async def start(self):
        """Start the radar server components"""
        self.running = True
        
        try:
            # 1. Initialize radar hardware
            self.logger.debug("Initializing radar hardware...")
            self.radar = BGT60TR13C()
            if self.config_file:
                from ..config import RadarConfig
                import os
                if os.path.exists(self.config_file):
                    self.logger.debug(f"Loading configuration from {self.config_file}")
                    radar_config = RadarConfig.read_yaml(self.config_file)
                    self.radar.configure(radar_config)
                else:
                    self.logger.warning(f"Config file {self.config_file} not found, using defaults")
                    self.radar.configure()
            else:
                self.radar.configure()
            
            self.controller = create_adaptive_controller(self.radar)
            
            # 2. Initialize ZMQ Streamer
            self.logger.debug(f"Initializing ZMQ streamer on ports {self.port} (data) and {self.status_port} (status)...")
            self.streamer = RadarStreamer(
                data_port=self.port,
                status_port=self.status_port,
                host="0.0.0.0")
            
            # 3. Initialize HTTP API
            self.logger.debug(f"Initializing HTTP API on port {self.http_port}...")
            app = create_app(self.radar, self.controller)
            
            # Configure uvicorn logging to match our format
            # Only show "Uvicorn running" message at INFO, rest at DEBUG
            log_config = {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "default": {
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        "datefmt": "%Y-%m-%d %H:%M:%S"
                    },
                    "access": {
                        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        "datefmt": "%Y-%m-%d %H:%M:%S"
                    }
                },
                "filters": {
                    "uvicorn_filter": {
                        "()": uvicorn_filter_factory
                    }
                },
                "handlers": {
                    "default": {
                        "formatter": "default",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stdout",
                        "filters": ["uvicorn_filter"]
                    },
                    "access": {
                        "formatter": "access",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stdout"
                    }
                },
                "loggers": {
                    "uvicorn": {
                        "handlers": ["default"],
                        "level": "DEBUG",
                        "propagate": False
                    },
                    "uvicorn.error": {
                        "handlers": ["default"],
                        "level": "DEBUG",
                        "propagate": False
                    },
                    "uvicorn.access": {
                        "handlers": ["access"],
                        "level": "DEBUG",
                        "propagate": False
                    }
                }
            }
            
            config = uvicorn.Config(
                app=app, 
                host="0.0.0.0", 
                port=self.http_port, 
                log_level="info",
                log_config=log_config
            )
            self.api_server = uvicorn.Server(config)
            
            # 4. Initialize Discovery Announcer
            self.logger.debug("Initializing discovery service...")
            self.announcer = RadarAnnouncer(
                port=self.port,
                name=f"PiRadar-{socket.gethostname()}",
                properties={
                    "data_port": self.port,
                    "http_port": self.http_port,
                    "status_port": self.status_port,
                    "version": "1.1.0",
                    "api_type": "http"
                }
            )
            await self.announcer.start()
            
            # 5. Run loops
            # We need to run the API server and the radar data loop concurrently
            
            server_task = asyncio.create_task(self.api_server.serve())
            radar_task = asyncio.create_task(self._radar_loop())
            
            self.logger.debug("Server started successfully")
            
            # Wait for either task to complete (or be cancelled)
            done, pending = await asyncio.wait(
                [server_task, radar_task], 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # If we get here, one task has finished. Cancel the others.
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
            # Check for exceptions in done tasks (but ignore CancelledError)
            for task in done:
                try:
                    if task.exception() and not isinstance(task.exception(), (asyncio.CancelledError, KeyboardInterrupt)):
                        self.logger.error(f"Task failed: {task.exception()}")
                        raise task.exception()
                except AttributeError:
                    # Task might not have exception() method in some cases
                    pass

        except (asyncio.CancelledError, KeyboardInterrupt):
            self.logger.debug("Server cancelled")
        except Exception as e:
            self.logger.error(f"Server failed: {e}", exc_info=True)
        finally:
            await self.stop()

    async def _radar_loop(self):
        """Main radar data acquisition and streaming loop"""
        self.logger.debug("Starting radar data loop")
        
        # Auto-start radar
        if not self.radar.is_running:
            self.radar.start()
        
        last_status_time = time.time()
        import queue
        from functools import partial
        
        while self.running:
            try:
                if self.radar.is_running:
                    # Get frame from radar's frame_buffer queue
                    # Use asyncio.to_thread to run the blocking queue.get() in a thread
                    try:
                        frame_data = await asyncio.to_thread(
                            partial(self.radar.frame_buffer.get, timeout=0.1)
                        )
                        
                        if frame_data is not None:
                            # Publish frame via ZMQ
                            await self.streamer.publish_frame(frame_data)
                            
                            # Periodically publish status (every 5 seconds or every 60 frames)
                            current_time = time.time()
                            if (current_time - last_status_time >= 5.0) or (self.streamer.frame_count % 60 == 0):
                                status_str = "Running" if self.radar.is_running else "Stopped"
                                await self.streamer.publish_status(status_str)
                                last_status_time = current_time
                                
                    except queue.Empty:
                        # No frame available, continue
                        await asyncio.sleep(0.001)
                        continue
                            
                else:
                    # Radar is stopped, just sleep a bit
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"Error in radar loop: {e}", exc_info=True)
                await asyncio.sleep(1.0) # Backoff on error

    async def stop(self):
        """Stop all components"""
        if not self.running:
            return  # Already stopped
            
        self.running = False
        self.logger.debug("Stopping server components...")
        
        # Stop announcer first to unregister service
        if self.announcer:
            try:
                await self.announcer.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping announcer: {e}")
            
        # Stop radar hardware
        if self.radar:
            try:
                self.radar.stop()
                # BGT60TR13C doesn't have a close() method, cleanup is handled by context manager
                # or via stop() which cleans up the data collection thread
            except Exception as e:
                self.logger.warning(f"Error stopping radar: {e}")
                
        # Close ZMQ streamer
        if self.streamer:
            try:
                await self.streamer.close()
            except Exception as e:
                self.logger.warning(f"Error closing streamer: {e}")
            
        # Cancel any remaining tasks
        try:
            # Get all tasks in the current event loop
            loop = asyncio.get_event_loop()
            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
            if tasks:
                self.logger.debug(f"Cancelling {len(tasks)} remaining tasks...")
                for task in tasks:
                    task.cancel()
                # Wait a bit for tasks to cancel, but don't wait too long
                await asyncio.wait(tasks, timeout=0.5, return_when=asyncio.ALL_COMPLETED)
        except Exception as e:
            self.logger.debug(f"Error cancelling tasks: {e}")
        
        self.logger.debug("Server stopped")

# Signal handling helper
def install_signal_handlers(loop):
    def handle_stop():
        for task in asyncio.all_tasks(loop):
            task.cancel()
            
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_stop)

