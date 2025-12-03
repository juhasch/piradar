"""Test script to see if changing modes works"""

import json
import logging
import numpy as np
import yaml
from piradar.hw import BGT60TR13C
import time
from piradar.config import RadarConfig

# Configure logging to debug level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_radar_data_collection():
    num_frames = 10

    radar_config = RadarConfig.read_yaml("radar_config.yaml")

    with BGT60TR13C() as radar:
        radar.soft_reset(radar.RESET_SW)
        radar.configure(radar_config)
        radar.start()
        
        frames = []
        t0 = time.time()
        for i in range(num_frames):
            frames.append(radar.frame_buffer.get())
            dt = time.time() - t0
            t0 = time.time()
            logger.info(f'Frame {i} received in {dt*1e3:.2f} ms')
            logger.debug(f'Received frame {i}')
        radar.stop()
        time.sleep(1)
        #radar.configure(radar_config)
        radar.start()
        
        frames = []
        t0 = time.time()
        for i in range(num_frames):
            frames.append(radar.frame_buffer.get())
            dt = time.time() - t0
            t0 = time.time()
            logger.info(f'Frame {i} received in {dt*1e3:.2f} ms')
            logger.debug(f'Received frame {i}')
        
        radar.stop()

        data = np.array(frames)
        logger.info(f'Data shape: {data.shape}')
        np.save(f"frames.npy", data.astype(np.float32))
        logger.info("Frames saved to frames.npy")
          
if __name__ == "__main__":
    test_radar_data_collection()
