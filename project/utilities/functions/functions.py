"""providing functions"""
import numpy as np

def softmax(x):
    """softmax"""
    max_val = np.max(x, axis = 1, keepdims=True)
    exp_x = np.exp(x - max_val)
    sum_of_exp_x = np.sum(exp_x, axis = 1, keepdims=True)
    return exp_x / sum_of_exp_x

def cross_entropy_error(x, t):
    """t : one hot encoding label"""
    entropy_value = x * t
    result = np.sum(entropy_value, axis = 1, keepdims=True)
    return -1 * np.log(result)
