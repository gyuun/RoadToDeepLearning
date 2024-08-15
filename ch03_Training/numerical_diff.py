"""Providing gradient function"""
import numpy as np

def gradient(f, x):
    """f에 대해 x 에서의 수치미분 연산.

    parameter : (f : 미분할 함수), (x : ndarray)
    return : ndarray
    """
    h = 1e-4
    grad = np.zeros_like(x)
    if x.ndim == 2: # 가중치
        for i in range(len(x)):
            for idx in range(len(x[0])):
                tmp_val = x[i, idx]
                x[i, idx] = tmp_val + h
                fxh1 = f(x)
                x[i, idx] = tmp_val - h
                fxh2 = f(x)
                grad[i, idx] = (fxh1 - fxh2) / (2*h)
                x[i, idx] = tmp_val
    elif x.ndim == 1: # 편향
        for i in range(x.size):
            tmp_val = x[i]
            x[i] = tmp_val + h
            fxh1 = f(x)
            x[i] = tmp_val - h
            fxh2 = f(x)
            grad[i] = (fxh1 - fxh2) / (2*h)
            x[i] = tmp_val
    return grad
