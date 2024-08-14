"""Train mnist dataset with numerical diff, gradient_descent"""
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
import numpy as np
from ch03_Training.two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize= True, one_hot_label= True)
    """ x_train (60000, 784), t_train (60000, 10)
    x_test (10000, 784) t_test (10000, 10)
    """
    train_loss_list = []

    REPETITION = 10000
    train_size = x_train.shape[0]
    BATCHSIZE = 100

    LEARNINGRATE = 0.1
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size = 10)

    for i in range(REPETITION):
        batch_mask = np.random.choice(train_size, BATCHSIZE)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.numerical_gradient(x_batch, t_batch) # x_batch (100, 784) t_batch (100, 10)

        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= LEARNINGRATE * grad[key]

        loss = network.loss(x_batch, t_batch)

        train_loss_list.append(loss)
