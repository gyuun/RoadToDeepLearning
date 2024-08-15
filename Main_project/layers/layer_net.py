"""Providng Layer class"""
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
import numpy as np
from Main_project.layers.layer import Affine, ReLu, SoftmaxWithLoss

class Layer:
    """3 hidden layer net object"""
    def __init__( # pylint: disable=too-many-arguments
            self, input_size=784,
            first_hidden_layer = 100, second_hidden_layer = 50,
            third_hidden_layer = 50, output= 10):
        """가중치 초기화 : He 초깃값, 계층 초기화"""
        self.params= {}
        self.params['W1'] = np.sqrt(2/input_size) \
                        * np.random.randn(input_size, first_hidden_layer)
        self.params['W2'] = np.sqrt(2/first_hidden_layer) \
                        * np.random.randn(first_hidden_layer, second_hidden_layer)
        self.params['W3'] = np.sqrt(2/second_hidden_layer) \
                        * np.random.randn(second_hidden_layer, third_hidden_layer)
        self.params['W4'] = np.sqrt(2/third_hidden_layer) \
                        * np.random.randn(third_hidden_layer, output)
        self.params['b1'] = np.zeros(first_hidden_layer)
        self.params['b2'] = np.zeros(second_hidden_layer)
        self.params['b3'] = np.zeros(third_hidden_layer)
        self.params['b4'] = np.zeros(output)

        self.layers = {}
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['ReLu1'] = ReLu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLu2'] = ReLu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['ReLu3'] = ReLu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()

        self.t = None
        self.predict_val = None

    def predict(self, x):
        """순전파 연산. ***Python 3.7부터 딕셔너리에 순서 존재"""
        for layer in self.layers.values():
            x = layer.forward(x)
        self.predict_val = x
        return x

    def loss(self, x, t, lambda_=0.01):
        """손실함수. 가중치 감소법 적용을 위한 L2 노름"""
        self.t = t
        predict_res = self.predict(x)
        res = self.last_layer.forward(predict_res, t)
        norm = 0
        for w in self.params.values():
            norm += np.sum(w**2)
        loss_val = np.sum(res)/len(res) + (0.5 * lambda_ * norm)

        return loss_val

    def accuracy(self):
        """미니배치에 대한 모델 정확도 평가"""
        y = self.predict_val
        y = np.argmax(y, axis = 1)
        t = np.argmax(self.t, axis = 1)
        accuracy = np.sum(y == t) / len(y)
        return accuracy

    def gradient(self):
        """역전파 연산. ***Python 3.7부터 딕셔너리에 순서 존재"""
        dout = self.last_layer.backward()
        reversed_layer = reversed(list(self.layers.values()))
        for layer in reversed_layer:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dw
        grads['b3'] = self.layers['Affine3'].db
        grads['W4'] = self.layers['Affine4'].dw
        grads['b4'] = self.layers['Affine4'].db

        return grads
