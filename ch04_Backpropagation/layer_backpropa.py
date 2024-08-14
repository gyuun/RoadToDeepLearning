"""Providing TwoLayerNet Class"""
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
from collections import OrderedDict
import numpy as np
from ch04_Backpropagation.function_layer import ReLu, Affine, SoftmaxWithLoss

class TwoLayerNet:
    """TwoLayerNet class with backpropagation"""
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std = 0.01):
        """initialize weights, bias and layers"""
        self.params = {}
        self.params['W1'] = weight_init_std \
                            * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std \
                            * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLu()
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        """Propagation"""
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """출력층 순전파"""
        y = self.predict(x)

        return self.last_layer.forward(y,t)

    def accuracy(self, x, t):
        """배치 정확도 평가"""
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1:
            t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def gradient(self, x, t):
        """역전파 알고리즘으로 그래디언트 연산"""
        self.loss(x,t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads
