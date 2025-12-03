from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True, cache=True)
def _numba_process_uint12(data):
    """
    Internal Numba function to process uint12 data.
    """
    num_groups = data.shape[0] // 3
    result = np.empty(2 * num_groups, dtype=np.float32)
    
    for i in prange(num_groups):
        # Extract bytes for this group
        idx = i * 3
        b0 = data[idx]
        b1 = data[idx + 1] 
        b2 = data[idx + 2]
        
        # Calculate 12-bit values
        first_12bit = (b0 << 4) | (b1 >> 4)
        second_12bit = ((b1 & 0x0F) << 8) | b2
        
        # Store results
        result[i * 2] = first_12bit
        result[i * 2 + 1] = second_12bit
    
    return result

def read_uint12(data_chunk):
    """
    Numba-optimized version of read_uint12.
    
    Args:
        data_chunk: Raw bytes containing packed 12-bit data.
                    Length must be divisible by 3.
    
    Returns:
        numpy.ndarray: Array of 12-bit values as float32
    """
    if len(data_chunk) % 3 != 0:
        raise ValueError("Data chunk length must be divisible by 3")
    
    data = np.frombuffer(data_chunk, dtype=np.uint8)
    return _numba_process_uint12(data)
