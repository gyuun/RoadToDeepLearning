"""evaluate trained weights"""
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
import numpy as np
from dataset.mnist import load_mnist
from project.model.model import FiveLayerNeuralNetwork

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize= True, one_hot_label= True)
    """we'll use x_test, t_test data"""
    total_accuracy = 0 #pylint: disable=invalid-name
    REPETITION = 100
    BATCHSIZE = 100
    network = FiveLayerNeuralNetwork(
            input_size=784,
            first_hidden_layer = 100, second_hidden_layer = 50,
            third_hidden_layer = 50, output= 10)
    network.load_weights()

    for i in range(REPETITION):
        batch_mask = np.random.choice(len(x_test), BATCHSIZE)
        x_batch = x_test[batch_mask]
        t_batch = t_test[batch_mask]
        loss = network.loss(x_batch, t_batch)
        total_accuracy = total_accuracy + network.accuracy()*100

    print(f'Model\'s total_accuracy about test data : {total_accuracy / REPETITION}')
