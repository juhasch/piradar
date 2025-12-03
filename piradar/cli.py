"""
Command-line interface for PiRadar
Provides easy access to radar functionality from the command line
"""

import argparse
import sys
import os
import time
import signal
import logging
import queue
import asyncio
from typing import Optional
import psutil
from tqdm import tqdm

from piradar.hw import BGT60TR13C, BGT60TR13CError
from piradar.hw.discovery import RadarScanner
from piradar.config import RadarConfig
# New Server Import
from piradar.hw.server import RadarServer


def setup_logging(level: str) -> None:
    """Set up logging configuration"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\nShutting down...')
    # For async code (serve command), we let uvicorn handle shutdown
    # For sync code, we can exit directly
    # This handler is mainly for non-async commands
    sys.exit(0)

def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB"""
    if psutil is None:
        return 0.0
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0

def check_chip_id(args) -> None:
    """Check and display chip ID"""
    print("Checking chip ID...")
    
    with BGT60TR13C() as radar:
        radar.check_chip_id()
        print("✓ Chip identified as BGT60TR13C")

def list_radars(args) -> None:
    """List discovered radars on the network"""
    print(f"Scanning for radars (timeout: {args.timeout}s)...")
    
    scanner = RadarScanner()
    radars = scanner.scan(timeout=args.timeout)
    
    if not radars:
        print("No radars found.")
        return
        
    print(f"\nFound {len(radars)} radar(s):")
    print("=" * 80)
    print(f"{'Name':<25} | {'IP Address':<15} | {'Ports (Data/HTTP/Sts)':<20} | {'Version':<10}")
    print("-" * 80)
    
    for radar in radars:
        # Discovery might return old style ports or new style properties
        # Adapt display accordingly
        if hasattr(radar, 'properties') and 'http_port' in radar.properties:
            ports = f"{radar.data_port}/{radar.properties['http_port']}/{radar.status_port}"
        else:
            # Fallback or old version assumption
            ports = f"{radar.data_port}/{radar.command_port}/{radar.status_port}"
            
        print(f"{radar.name:<25} | {radar.address:<15} | {ports:<20} | {radar.version:<10}")
    print("=" * 80)


def monitor_data(args) -> None:
    """Monitor radar data in real-time"""
    print(f"Starting data monitoring for {args.duration} seconds...")
    print("Press Ctrl+C to stop early")
    
    try:
        with BGT60TR13C() as radar:
            # Configure radar (use provided config or default)
            if args.config and os.path.exists(args.config):
                radar_config = RadarConfig.read_yaml(args.config)
                radar.configure(radar_config)
            else:
                radar.configure()  # Use default configuration
            radar.set_fifo_parameters()
            
            # Start data collection
            radar.start()
            
            start_time = time.time()
            frame_count = 0
            last_frame_time = start_time
            
            print("\n" + "="*60)
            print("RADAR DATA MONITORING")
            print("="*60)
            print("Frame # | Frame Rate (FPS) | Time Elapsed | Status")
            print("-"*60)
            
            while time.time() - start_time < args.duration:
                try:
                    frame = radar.frame_buffer.get(timeout=1.0)
                    frame_count += 1
                    current_time = time.time()
                    elapsed = current_time - start_time
                    
                    # Calculate frame rate
                    if frame_count > 1:
                        time_since_last = current_time - last_frame_time
                        fps = 1.0 / time_since_last if time_since_last > 0 else 0
                    else:
                        fps = 0
                    
                    last_frame_time = current_time
                    
                    print(f"{frame_count:7d} | {fps:13.1f} | {elapsed:11.1f}s | Data OK")
                    
                except queue.Empty:
                    print(f"{'':7s} | {'':13s} | {time.time() - start_time:11.1f}s | No frames...")
                    continue
            
            radar.stop()
            print(f"\nMonitoring complete: {frame_count} frames received")
            
    except KeyboardInterrupt:
        print("\nStopped by user")

def record_data(args) -> None:
    """Record radar data to file"""
    print(f"Recording {args.frames} frames...")
    
    # Generate output filename if not provided
    if not args.output:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = f"radar_data_{timestamp}.{args.format}"
    
    print(f"Output file: {args.output}")
    
    try:
        with BGT60TR13C() as radar:
            # Configure radar (use provided config or default)
            if args.config and os.path.exists(args.config):
                radar_config = RadarConfig.read_yaml(args.config)
                radar.configure(radar_config)
            else:
                radar.configure()  # Use default configuration
            radar.set_fifo_parameters()
            
            # Start data collection
            radar.start()
            
            frames = []
            frame_count = 0
            
            print("Collecting frames...")
            if tqdm is not None:
                with tqdm(total=args.frames, desc="Recording") as pbar:
                    while frame_count < args.frames:
                        try:
                            frame = radar.frame_buffer.get(timeout=5.0)
                            frames.append(frame)
                            frame_count += 1
                            pbar.update(1)
                                
                        except queue.Empty:
                            print("Timeout waiting for frame, continuing...")
                            continue
                        except (ValueError, TypeError) as e:
                            print(f"Error processing frame data: {e}")
                            continue
            else:
                while frame_count < args.frames:
                    try:
                        frame = radar.frame_buffer.get(timeout=5.0)
                        frames.append(frame)
                        frame_count += 1
                        
                        if frame_count % 10 == 0:
                            print(f"  Collected {frame_count}/{args.frames} frames")
                            
                    except queue.Empty:
                        print("Timeout waiting for frame, continuing...")
                        continue
                    except (ValueError, TypeError) as e:
                        print(f"Error processing frame data: {e}")
                        continue
            
            radar.stop()
            print(f"Data collection complete: {frame_count} frames")
            
            # Save data
            if args.format == 'npy':
                import numpy as np
                data = np.array(frames)
                np.save(args.output, data.astype(np.float32))
                print(f"Saved {data.shape} to {args.output}")
                
            elif args.format == 'h5':
                import h5py
                with h5py.File(args.output, 'w') as f:
                    # Create datasets
                    f.create_dataset('frames', data=frames)
                    
                    # Add metadata if requested
                    if args.metadata:
                        f.attrs['timestamp'] = time.time()
                        f.attrs['frame_count'] = frame_count
                        f.attrs['format'] = 'h5'
                        if args.config:
                            f.attrs['config_file'] = args.config
                
                print(f"Saved {len(frames)} frames to {args.output}")
            
    except KeyboardInterrupt:
        print("\nStopped by user")

def publish_data(args) -> None:
    """Publish radar data via ZeroMQ (Legacy/Basic)"""
    # Note: The serve command is preferred as it offers the HTTP API.
    # This is a basic publisher loop using the new RadarStreamer if we wanted to use it,
    # but for now keeping the simple stdout debug or implementing a simple ZMQ pub.
    
    from piradar.hw import RadarStreamer
    
    print(f"Starting basic ZeroMQ publisher on port {args.port}...")
    print("Press Ctrl+C to stop...")
    
    try:
        # Create streamer
        streamer = RadarStreamer(data_port=args.port, status_port=args.port+2)
        
        async def run_publisher():
            with BGT60TR13C() as radar:
                if args.config and os.path.exists(args.config):
                    radar_config = RadarConfig.read_yaml(args.config)
                    radar.configure(radar_config)
                else:
                    radar.configure()
                radar.set_fifo_parameters()
                radar.start()
                
                frame_count = 0
                start_time = time.time()
                
                print("Publishing frames...")
                while True:
                    if args.frames and frame_count >= args.frames:
                        print(f"Reached frame limit: {args.frames}")
                        break
                    
                    try:
                        # Get frame from radar (blocking in this sync/async mix context, but ok for cli)
                        frame = radar.frame_buffer.get(timeout=1.0)
                        frame_count += 1
                        
                        # Publish
                        await streamer.publish_frame(frame)
                        
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        
                        if frame_count % 10 == 0:
                            print(f"Frame {frame_count}: {len(frame)} bytes ({fps:.1f} FPS)")
                        
                    except queue.Empty:
                        continue
                
                radar.stop()
        
        asyncio.run(run_publisher())
            
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")

async def serve_async(args) -> None:
    """Run radar server with both basic and advanced capabilities"""
    print("Press Ctrl+C to stop...")
    
    server = RadarServer(
        port=args.port,
        config_file=args.config,
        synthetic=args.synthetic
    )
    
    try:
        await server.start()
    except KeyboardInterrupt:
        # KeyboardInterrupt is handled by uvicorn's signal handling
        # but we catch it here to ensure clean shutdown
        pass
    except asyncio.CancelledError:
        # Task cancellation is expected during shutdown
        pass

def serve(args) -> None:
    """Run radar server"""
    try:
        asyncio.run(serve_async(args))
    except KeyboardInterrupt:
        # This should rarely be reached since uvicorn handles signals,
        # but it's here as a safety net
        print("\nServer stopped by user")
    except SystemExit as e:
        # SystemExit(0) is a clean shutdown from uvicorn's signal handling
        # We suppress the traceback for clean exits
        if e.code != 0:
            raise
        # For exit code 0, we just exit silently
        sys.exit(0)

def main() -> None:
    """Main CLI entry point"""
    # Parent parser for common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default=argparse.SUPPRESS,
        help='Logging level (default: INFO)'
    )

    parser = argparse.ArgumentParser(
        description="PiRadar - BGT60TR13C radar sensor driver CLI",
        parents=[parent_parser],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s check-id                                    # Check chip identification
  %(prog)s monitor --duration 30                      # Monitor data for 30 seconds
  %(prog)s record --frames 100 --output data.npy      # Record 100 frames to file
  %(prog)s publish --port 6666                        # Publish data on port 6666
  %(prog)s serve                                       # Run radar server with hardware
  %(prog)s serve --config my_config.yaml             # Run with custom radar configuration
  %(prog)s serve --port 6666                         # Run radar server on port 6666
  %(prog)s --log-level DEBUG check-id                 # Check ID with debug logging
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check ID command
    check_parser = subparsers.add_parser('check-id', help='Check chip identification', parents=[parent_parser])

    # List command
    list_parser = subparsers.add_parser('list', help='List discovered radars on the network', parents=[parent_parser])
    list_parser.add_argument('--timeout', type=float, default=2.0, help='Scan duration in seconds (default: 2.0)')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor radar data in real-time', parents=[parent_parser])
    monitor_parser.add_argument('--duration', type=int, default=30, help='Monitoring duration in seconds (default: 30)')
    monitor_parser.add_argument('--config', type=str, help='Path to radar configuration file')
    
    # Record command
    record_parser = subparsers.add_parser('record', help='Record radar data to file', parents=[parent_parser])
    record_parser.add_argument('--frames', type=int, default=100, help='Number of frames to record (default: 100)')
    record_parser.add_argument('--output', type=str, help='Output filename (auto-generated if not specified)')
    record_parser.add_argument('--format', choices=['npy', 'h5'], default='npy', help='Output format (default: npy)')
    record_parser.add_argument('--config', type=str, help='Path to radar configuration file')
    record_parser.add_argument('--metadata', action='store_true', help='Include metadata in output file (HDF5 only)')
    
    # Publish command
    publish_parser = subparsers.add_parser('publish', help='Stream radar data via ZeroMQ (no control API)', parents=[parent_parser])
    publish_parser.add_argument('--port', type=int, default=5555, help='ZeroMQ publisher port (default: 5555)')
    publish_parser.add_argument('--config', type=str, help='Path to radar configuration file')
    publish_parser.add_argument('--frames', type=int, help='Maximum frames to publish (default: unlimited)')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Run radar server with HTTP API and ZMQ streaming', parents=[parent_parser])
    serve_parser.add_argument('--synthetic', action='store_true', help='Use synthetic data instead of real radar hardware (default: hardware)')
    serve_parser.add_argument('--port', type=int, default=5555, help='Base port for ZMQ services (default: 5555)')
    serve_parser.add_argument('--config', type=str, help='Path to radar configuration file')
    
    args = parser.parse_args()
    
    setup_logging(getattr(args, 'log_level', 'INFO'))
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Only register signal handler for non-async commands
    # The 'serve' command uses uvicorn which handles signals itself
    if args.command != 'serve':
        signal.signal(signal.SIGINT, signal_handler)
    
    # Execute command
    try:
        if args.command == 'check-id':
            check_chip_id(args)
        elif args.command == 'list':
            list_radars(args)
        elif args.command == 'monitor':
            monitor_data(args)
        elif args.command == 'record':
            record_data(args)
        elif args.command == 'publish':
            publish_data(args)
        elif args.command == 'serve':
            serve(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except (BGT60TR13CError, RuntimeError, OSError) as e:
        print(f"Radar Error: {e}")
        sys.exit(1)
    except (ValueError, TypeError) as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Import Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
