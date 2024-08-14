"""Providing momentum, Adagrad, Adam class"""
import numpy as np

class Momentum: # pylint: disable=too-few-public-methods
    """momentum 클래스"""
    def __init__(self, params):
        """초깃값 설정"""
        self.alpha = 0.9
        self.lr = 0.01
        self.v = {}
        for key in params.keys():
            self.v[key] = np.zeros_like(params[key])

    def update(self, params ,grads):
        """모멘텀을 이용한 가중치 조작"""
        for key in grads.keys():
            self.v[key] = (self.alpha * self.v[key]
                            - self.lr * grads[key])
            params[key] = params[key] + self.v[key]

        return params


class Adagrad: # pylint: disable=too-few-public-methods
    """Adagrad 클래스"""
    def __init__(self, params):
        """초깃값 설정"""
        self.lr = 0.01
        self.h = {}
        for key in params.keys():
            self.h[key] = np.zeros_like(params[key])

    def update(self, params, grads):
        """가중치 업데이트"""
        for key in grads.keys():
            self.h[key] = self.h[key] + grads[key]*grads[key]
            params[key] = params[key] - self.lr * 1/self.h**(1/2) * grads[key]


class Adam: # pylint: disable=too-few-public-methods
    """Adam 클래스"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """초깃값 설정"""
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        """가중치 업데이트"""
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)


class DropOut:
    """드롭아웃"""
    def __init__(self, drouput_ratio = 0.5):
        """드롭아웃 비율 설정"""
        self.dropout_ratio = drouput_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        """순전파"""
        if train_flg:
            self.mask = np.rand(*x.shape) > self.dropout_ratio
            return x * self.mask

        return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        """역전판"""
        return dout * self.mask
