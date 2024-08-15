"""Providing TwoLayerNet class"""
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
import numpy as np
from ch03_Training.numerical_diff import gradient
from ch02_Network.network import sigmoid, softmax

class TwoLayerNet:
    """2층 신경망 클래스"""      
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std= 0.01):
        """임의의 가중치, 편향 값을 만든다.

        parameter : 
        (input_size : 입력층의 뉴런 수)
        (hidden_size : 은닉층의 뉴런 수)
        (output_size : 출력층의 뉴런 수)
        """
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """가중치, 편향을 통해 출력값을 반환한다.

        parameter : x (batch_size, input_size)

        return : y (batch_size, 10)
        """
        w1, w2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """미니배치 학습에서 손실함수는 평균이다.

        parameter : (x : 2darray), (t : 2darray)
        """
        y = self.predict(x)
        y = np.argmax(y, axis = 1) # axis = 1 : 열을 따라색인을 찾음.
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(len(x))
        return accuracy

    def numerical_gradient(self, x, t):
        """현재 가중치와 편향의 기울기를 반환한다.

        parameter : (x : 입력 값), (t : 정답 레이블)
        """

        grads = {}
        grads['W1'] = gradient(self.loss(x,t), self.params['W1']) # W1은 2차원 데이터 784, 50
        grads['b1'] = gradient(self.loss(x,t), self.params['b1'])
        grads['W2'] = gradient(self.loss(x,t), self.params['W2'])
        grads['b2'] = gradient(self.loss(x,t), self.params['b2'])

        return grads
