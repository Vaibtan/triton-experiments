#!usr/bin/bash/python3

# attorch implementation
from typing import List, Optional
import torch as T
import triton

def allow_tf32() -> bool: return T.cuda.get_device_capability()[0] >= 8

def get_n_stages(n_stages: int = 2) -> int: 
    return 2 if T.cuda.get_device_capability()[0] < 8 else n_stages


def get_output_dtype(input_dtype: T.dtype = T.float32, \
    autocast: Optional[str] = None) -> T.dtype:
    # returns apt output dtype for auto mixed precision given input dtype & op's autocast behaviour
    dtype = T.get_autocast_dtype('cuda')
    assert dtype, f'Only autocast to float16 is supported, received {dtype}'
    if T.is_autocast_enabled():
        if autocast is None: return input_dtype
        elif autocast == 'fp16': return T.float16
        elif autocast == 'fp32': return T.float32
        else: raise RuntimeError(f'Autocast type {autocast} is invalid. '\
            'Options are None, fp16, and fp32')
    else: return input_dtype


def element_wise_kernel_configs(block_name: str = 'BLOCK_SIZE') -> List[triton.Config]:
    # returns kernel configs for element-wise ops
    # block_name <- block argument rows are distributed over
    return [triton.Config({block_name: 64}, num_warps = 2),
            triton.Config({block_name: 128}, num_warps = 2),
            triton.Config({block_name: 256}, num_warps = 4),
            triton.Config({block_name: 512}, num_warps = 4),
            triton.Config({block_name: 1024}, num_warps = 4)]


def warps_kernel_configs() -> List[triton.Config]:
    # kernel config w/ all possible #warps
    return [triton.Config({}, num_warps = 2 ** i) for i in range(6)]