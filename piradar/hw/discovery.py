"""
Radar Discovery Module using Zeroconf (mDNS/DNS-SD)
Handles announcing the radar on the network and discovering available radars.
"""

import socket
import logging
import asyncio
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from zeroconf import ServiceInfo, Zeroconf, ServiceBrowser, ServiceStateChange, IPVersion
from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf

logger = logging.getLogger(__name__)

def get_local_ip() -> str:
    """Attempt to find the local machine's IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

@dataclass
class DiscoveredRadar:
    name: str
    address: str
    data_port: int
    command_port: int
    status_port: int
    version: str = "unknown"
    hostname: str = ""

class RadarAnnouncer:
    def __init__(self, port: int, name: str = "PiRadar", properties: Optional[Dict[str, Any]] = None):
        self.zeroconf = None
        self.info = None
        self.port = port
        self.name = name
        self.properties = properties or {}

    async def start(self):
        """Announce the radar service on the network asynchronously."""
        ip_address = get_local_ip()
        service_type = "_piradar._tcp.local."
        service_name = f"{self.name}.{service_type}"
        
        logger.info(f"Announcing radar service at {ip_address}:{self.port}")
        
        encoded_props = {k: str(v) for k, v in self.properties.items()}
        unique_id = str(uuid.uuid4())[:8]
        unique_server_name = f"piradar-{unique_id}.local."

        self.info = ServiceInfo(
            service_type,
            service_name,
            addresses=[socket.inet_aton(ip_address)],
            port=self.port,
            properties=encoded_props,
            server=unique_server_name,
        )

        self.zeroconf = AsyncZeroconf()
        try:
            await self.zeroconf.async_register_service(self.info, allow_name_change=True)
            logger.debug(f"Service registered via Zeroconf as {self.info.name}")
        except Exception as e:
            logger.error(f"Failed to register service: {e!r}")

    async def stop(self):
        """Unregister the service asynchronously."""
        if self.zeroconf:
            try:
                if self.info:
                    await self.zeroconf.async_unregister_service(self.info)
            except Exception as e:
                logger.error(f"Error unregistering service: {e}")
            finally:
                try:
                    await self.zeroconf.async_close()
                    # Give a moment for background tasks to cleanup
                    # Zeroconf has background broadcast tasks that need time to finish
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.debug(f"Error closing zeroconf: {e}")
                finally:
                    self.zeroconf = None
                    logger.debug("Service unregistered")


class RadarScanner:
    def __init__(self):
        self.found_radars: Dict[str, DiscoveredRadar] = {}
        
    def _parse_service_info(self, zc: Zeroconf, type_: str, name: str) -> Optional[DiscoveredRadar]:
        info = zc.get_service_info(type_, name)
        if not info:
            return None
            
        # Parse addresses
        addresses = info.parsed_addresses()
        if not addresses:
            return None
        
        address = addresses[0]
        
        # Parse properties
        props = {}
        for k, v in info.properties.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            props[key] = val
            
        return DiscoveredRadar(
            name=name.replace("._piradar._tcp.local.", ""),
            address=address,
            data_port=int(props.get('data_port', 5555)),
            command_port=int(props.get('command_port', 5556)),
            status_port=int(props.get('status_port', 5557)),
            version=props.get('version', 'unknown'),
            hostname=info.server or ""
        )

    def scan(self, timeout: float = 2.0) -> List[DiscoveredRadar]:
        """
        Scan for radars on the network for a specified duration.
        Blocking call.
        """
        zeroconf = Zeroconf()
        
        def on_service_state_change(zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange) -> None:
            if state_change is ServiceStateChange.Added:
                radar = self._parse_service_info(zeroconf, service_type, name)
                if radar:
                    self.found_radars[name] = radar

        browser = ServiceBrowser(zeroconf, "_piradar._tcp.local.", handlers=[on_service_state_change])
        
        import time
        time.sleep(timeout)
        
        zeroconf.close()
        return list(self.found_radars.values())

    async def async_scan(self, timeout: float = 2.0) -> List[DiscoveredRadar]:
        """Async version of scan"""
        aiozc = AsyncZeroconf()
        found = {}
        
        def on_service_state_change(zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange) -> None:
            if state_change is ServiceStateChange.Added:
                # We need to use the synchronous zeroconf instance provided in the callback
                # or query it asynchronously. Since we are inside the callback, we might just store the name
                # and resolve later or try to resolve immediately.
                # Note: AsyncServiceBrowser behavior can be tricky to capture all info immediately 
                # without resolving.
                pass

        # Simpler approach for async: use the synchronous browser in a thread or just wrap the sync scan
        # since Zeroconf is primarily threaded. 
        # But let's try to use AsyncServiceBrowser properly if possible, or fallback to wrapping sync.
        
        # For simplicity and reliability given python-zeroconf usage patterns, 
        # wrapping the sync scan in run_in_executor is often safest for one-shot CLI commands.
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.scan, timeout)

