import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.mnist import load_mnist
from network import sigmoid, softmax, ReLu
import pickle
import numpy as np

def get_data():
    (x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten= True, normalize=False, one_hot_label=0)
    return x_test, t_test

def init_network():
    """주어진 가중치파일에서 가중치들을 읽어 온다"""
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'BottomToDeepLearning/Network/sample_weight.pkl')
    
    with open(file_path, 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

accuracy_cnt =0
batch_size = 100

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1) # argmax -> 가장 값이 큰 원소의 인덱스 반환 
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt)/len(x)))
    

