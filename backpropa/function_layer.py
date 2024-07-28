import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Training.loss_function import cross_entropy_error
from Network.network import softmax
import numpy as np

class MulLayer:
    """곱셉 노드 클래스
    
    forward : 순전파 메소드
    인스턴스 변수 초기화

    backward : 역전파 메소드
    return : xy를 바꿔서 입력값에 곱한 값
    """
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x 
        self.y = y
    
        return x*y
    
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class AddLayer:
    """덧셈 노드 클래스"""
    def __init__(self):
        pass 

    def forward(self, x, y):
        return x+y
    
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

class ReLu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0 # true 즉 0보다 작으면 0이다
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class Affine:
    def __init__(self, W, b):
        self.X = None
        self.W = W
        self.b = b
        self.dW = None
        self.db = None
    
    def forward(self, X):
        self.X = X
        out = np.dot(X, self.W)
        
        return out
    
    def backward(self, dout):
        dX = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis = 0)

        return dX

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss
    
    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
