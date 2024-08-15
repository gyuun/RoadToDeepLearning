"""mnist 데이터셋 학습 수행
: 3계층 신경망 클래스 
    : 입력층 784 1층 100 2층 50 3층 50 출력층 10
: 100개 인풋 미니배치 학습
: 수치미분과 오차역전파 사용
: 활성화 함수 Relu
: 출력층 활성화 함수 Softmax
: 손실함수 교차엔트로피합 (가중치 감소)
Optimizer : 
    가중치 초기값 : He Value
    매개변수 갱신 : Adam
"""
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
import numpy as np
from layers.layer_net import Layer
from optimizer.adam import Adam
from dataset.mnist import load_mnist

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize= True, one_hot_label= True)
    """ x_train (60000, 784), t_train (60000, 10)
    x_test (10000, 784) t_test (10000, 10)
    """
    train_loss_list = []

    REPETITION = 10000
    BATCHSIZE = 100
    LEARNINGRATE = 0.001
    EPOCH= 600

    network = Layer(
            input_size=784,
            first_hidden_layer = 100, second_hidden_layer = 50,
            third_hidden_layer = 50, output= 10)
    optimizer = Adam(lr=LEARNINGRATE)
    for i in range(REPETITION):
        batch_mask = np.random.choice(len(x_train), BATCHSIZE)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        loss = network.loss(x_batch, t_batch)
        grad = network.gradient()
        optimizer.update(params=network.params, grads = grad)

    for i in range(100):
        batch_mask = np.random.choice(len(x_test), BATCHSIZE)
        x_batch = x_test[batch_mask]
        t_batch = t_test[batch_mask]
        loss = network.loss(x_batch, t_batch)
        accuracy = network.accuracy()
        if i%10 == 0:
            print(f'batch accuracy : {accuracy}')
            print(f'loss value: {loss}')
