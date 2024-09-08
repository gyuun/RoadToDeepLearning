"""Handmade padding process module using list not ndarray"""
import numpy as np

def padding(tensor : np.array, pad : int) -> np.array:
    """4dim array pading function"""
    n, c, h, w = tensor.shape
    result = np.zeros((n, c, h + 2*pad, w + 2*pad))
    result[:,:,pad:pad+h,pad:pad+w] = tensor

    return result
