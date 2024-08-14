"""Providing layer classes"""
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
import numpy as np
from ch02_Network.network import softmax
from ch03_Training.loss_function import cross_entropy_error

class MulLayer:
    """곱셉 노드 클래스
    
    forward : 순전파 메소드
    인스턴스 변수 초기화

    backward : 역전파 메소드
    return : xy를 바꿔서 입력값에 곱한 값
    """
    def __init__(self):
        """initialize"""
        self.x = None
        self.y = None

    def forward(self, x, y):
        """순전파"""
        self.x = x
        self.y = y

        return x*y

    def backward(self, dout):
        """역전파 : 위아래가 바뀌어 곱해진다."""
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class AddLayer:
    """덧셈 노드 클래스"""
    def __init__(self):
        """initialize"""

    def forward(self, x, y):
        """순전파"""
        return x+y

    def backward(self, dout):
        """덧셈함수의 역전파는 1"""
        dx = dout * 1
        dy = dout * 1
        return dx, dy

class ReLu:
    """ReLu 활성화 함수 클래스"""
    def __init__(self):
        """리스트 마스킹 이용"""
        self.mask = None

    def forward(self, x):
        """순전파 : 마스킹 값이 true 즉 0보다 작으면 0이다"""
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        """역전파 : 출력값이 있는 원소만 전파"""
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:
    """sigmoid 계층"""
    def __init__(self):
        """역전파때 순전파의 출력 사용"""
        self.out = None

    def forward(self, x):
        """순전파"""
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        """역전파"""
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine:
    """Affine 계층"""
    def __init__(self, w, b):
        """Initialize"""
        self.x = None
        self.w = w
        self.b = b
        self.dw = None
        self.db = None

    def forward(self, x):
        """순전파"""
        self.x = x
        out = np.dot(x, self.w)

        return out

    def backward(self, dout):
        """역전파. 전치행렬 순서 주의"""
        dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)

        return dx

class SoftmaxWithLoss:
    """출력층과 손실함수 계층"""
    def __init__(self):
        """Initialize"""
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        """순전파"""
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self):
        """역전파"""
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
