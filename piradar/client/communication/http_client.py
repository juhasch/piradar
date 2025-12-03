"""
HTTP client for radar command communication.

Provides HttpClient and related utilities for sending commands to the radar server
via HTTP API.
"""

import logging
import time
import threading
from typing import Optional, Protocol

import httpx

from .common import resolve_host


class CommandClient(Protocol):
    """Protocol for command clients."""
    def send_command(self, command_id: str, parameters: Optional[dict] = None) -> Optional[dict]:
        """Send a command and return the response."""
        ...


class HttpClient:
    """
    Robust HTTP Client for radar commands.
    Replaces the deprecated ZMQ RobustRequestClient.
    """
    def __init__(self, host: Optional[str], port: int, timeout: float = 2.0):
        self.host = resolve_host(host)
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.timeout = timeout
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)
        logging.debug(f"Initialized HttpClient for {self.base_url}")

    def send_command(self, command_id: str, parameters: Optional[dict] = None) -> Optional[dict]:
        """Send command via HTTP POST."""
        payload = {
            "command_id": command_id,
            "parameters": parameters or {},
            "timestamp": time.time(),
            "source": "client"
        }
        
        try:
            response = self.client.post("/command", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # Convert HTTP response format to match what legacy ZMQ code expects
                # {success: bool, data: any, error_message: str}
                return data
            else:
                logging.error(f"HTTP Error {response.status_code}: {response.text}")
                return None
                
        except httpx.RequestError as e:
            logging.error(f"HTTP Request failed for command '{command_id}': {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error sending command '{command_id}': {e}")
            return None

    def close(self):
        """Close the HTTP client."""
        self.client.close()


def send_command(client: CommandClient, command_id: str, parameters: Optional[dict] = None) -> Optional[dict]:
    """Send a structured command via a command client.
    
    Args:
        client: CommandClient instance (e.g., HttpClient)
        command_id: Command name
        parameters: Optional parameters
        
    Returns:
        Response dict or None on failure
    """
    if hasattr(client, 'send_command'):
        return client.send_command(command_id, parameters)
    
    logging.error("Invalid client object passed to send_command")
    return None


def get_radar_config(client: CommandClient) -> dict:
    """Query radar configuration from server and return parsed config."""
    status_resp = send_command(client, "get_status", {})
    if not isinstance(status_resp, dict):
        return {}
    logging.debug(f"status_resp: {status_resp}")
    # Unwrap wrapped responses of the form {success, data, error_message}
    payload = None
    if "success" in status_resp:
        if status_resp.get("success") and isinstance(status_resp.get("data"), dict):
            payload = status_resp.get("data")
        else:
            err = status_resp.get("error_message")
            logging.warning("get_status failed: %s", err)
    else:
        payload = status_resp

    if isinstance(payload, dict):
        radar_cfg = payload.get("radar_config")
        if isinstance(radar_cfg, dict):
            # Log key radar configuration parameters if present
            cfg_parts = []
            for key in (
                "start_frequency_Hz",
                "stop_frequency_Hz",
                "chirp_duration_s",
                "frame_length",
                "frame_duration_s",
                "output_power",
                "adc_sample_rate_Hz",
            ):
                if key in radar_cfg:
                    cfg_parts.append(f"{key}={radar_cfg[key]}")
            if cfg_parts:
                logging.info("Radar config: %s", ", ".join(cfg_parts))
            return radar_cfg
    
    return {}


def start_radar_streaming(client: CommandClient) -> bool:
    """Send start command to radar server and return success status."""
    start_resp = send_command(client, "start", {})
    if isinstance(start_resp, dict) and start_resp.get("success", False):
        logging.info("Start acknowledged: %s", start_resp.get('data'))
        return True
    else:
        logging.warning("Start not acknowledged or failed.")
        return False


def stop_radar_streaming(client: CommandClient, retries: int = 3) -> bool:
    """Send stop command to radar server with retries and return success status."""
    for i in range(retries):
        stop_resp = send_command(client, "stop", {})
        if isinstance(stop_resp, dict) and stop_resp.get("success", False):
            logging.info("Stop acknowledged: %s", stop_resp.get('data'))
            return True
        
        logging.warning(f"Stop attempt {i+1}/{retries} failed or not acknowledged. Retrying...")
        time.sleep(0.5)
        
    logging.error("Stop failed after all retries.")
    return False


def send_keep_alive(client: CommandClient) -> bool:
    """Send keep-alive command to radar server to prevent timeout."""
    resp = send_command(client, "keep_alive", {})
    return resp is not None


class KeepAliveDaemon(threading.Thread):
    """Daemon thread to send keep-alive commands periodically."""
    def __init__(self, host: str, port: int, interval: float = 1.0):
        """
        Args:
            host: Server host
            port: HTTP command port (usually 5556)
            interval: Interval in seconds
        """
        super().__init__()
        self.host = resolve_host(host)
        self.port = port
        self.interval = interval
        self.running = False
        self.daemon = True
        self.client: Optional[HttpClient] = None

    def run(self):
        """Run the keep-alive daemon."""
        self.running = True
        # Create a dedicated client for this thread
        self.client = HttpClient(self.host, self.port, timeout=1.0)
        
        last_send = 0.0
        
        while self.running:
            now = time.time()
            if now - last_send >= self.interval:
                try:
                    # Use the dedicated keep_alive endpoint or generic command
                    self.client.client.post("/keep_alive")
                    last_send = now
                except httpx.RequestError:
                    logging.debug("Keep-alive failed (connection error)")
                except Exception as e:
                    logging.debug(f"Keep-alive exception: {e}")
            
            # Sleep in small chunks to allow quick stopping
            time.sleep(0.1)
            if not self.running:
                break
        
        if self.client:
            self.client.close()

    def stop(self):
        """Stop the keep-alive daemon."""
        self.running = False
        if self.is_alive():
            self.join(timeout=1.0)

