"""providing affine, relu, softmax layer"""
import os
import sys

import numpy as np
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
from project.utilities.functions.functions import softmax, cross_entropy_error

class Affine:
    """Affine 계층"""
    def __init__(self, w, b):
        """생성자로 가중치와 편향을 받는다."""
        self.x = None
        self.w = w
        self.b = b
        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """순전파"""
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dout):
        """역전파"""
        self.dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return self.dx

class ReLu:
    """ReLu 활성화 함수 계층"""
    def __init__(self):
        """리스트 마스킹 사용"""
        self.mask = None

    def forward(self, x):
        """순전파"""
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        """역전파 : 순전파 출력값이 존재하는 원소만 전파된다."""
        return dout * self.mask

class SoftmaxWithLoss:
    """출력 계층의 활성화 함수와 손실함수 계층"""
    def __init__(self):
        """y : softmax value
           t : one-hot-label data"""
        self.y = None
        self.t = None

    def forward(self, x, t):
        """result : 2ndim"""
        self.y = softmax(x)
        self.t = t
        result = cross_entropy_error(self.y, t)
        return result

    def backward(self):
        """역전파"""
        result = self.y - self.t
        return result
