"""Handmade padding process module using list not ndarray"""
import numpy as np

def padding(tensor : np.array, pad : int) -> np.array:
    """4dim array pading function"""
    h = tensor.shape[2]
    w = tensor.shape[3]
    result = np.zeros_like(tensor)
    result[:,:,pad:pad+h,pad:pad+w] = tensor[:,:,:,:]

    return result
