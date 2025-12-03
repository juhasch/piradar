"""Receive a preset number of frames and save them to a numpy array"""

import numpy as np
from piradar.hw import BGT60TR13C
from tqdm import tqdm


def test_radar_data_collection():
    """Test data collection functionality"""
    num_frames = 10

    with BGT60TR13C() as radar:
        radar.configure()
        radar.start()
        
        frames = []
        
        # Use tqdm for progress indication
        with tqdm(total=num_frames, desc="Collecting frames") as pbar:
            for _ in range(num_frames):
                frames.append(radar.frame_buffer.get())
                pbar.update(1)
        
        radar.stop()
        
        data = np.array(frames)
        print(f"Shape if data: {data.shape}")
        np.save(f"frames.npy", data.astype(np.float32))
          
if __name__ == "__main__":
    test_radar_data_collection()
