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
#pylint: disable=duplicate-code
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    train_accuracy_list =[]
    REPETITION = 100
#pylint: disable=duplicate-code
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
        train_loss_list.append(loss)

        accuracy = network.accuracy()
        train_accuracy_list.append(accuracy * 100)

        grad = network.gradient()
        optimizer.update(params=network.params, grads = grad)

    total_accuracy = 0 #pylint: disable=invalid-name
    for i in range(REPETITION):
        batch_mask = np.random.choice(len(x_test), BATCHSIZE)
        x_batch = x_test[batch_mask]
        t_batch = t_test[batch_mask]
        loss = network.loss(x_batch, t_batch)
        total_accuracy = total_accuracy + network.accuracy()*100

    print(f'Model\'s total_accuracy about test data : {total_accuracy / REPETITION}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.lineplot(
        ax = axes[0],x= list(range(0,REPETITION,1)),
        y=train_loss_list, markers='o')
    axes[0].set_xlabel('Repetition')
    axes[0].set_ylabel('loss')
    sns.lineplot(
        ax = axes[1], x= list(range(0,REPETITION,1)),
        y= train_accuracy_list, markers= 'o', color='red')
    axes[1].set_xlabel('Repetition')
    axes[1].set_ylabel('accuracy')
    plt.show()
