import numpy as np
def stair(x):
    if x<=0 :
        return 0
    else :
        return 1
def sigmoid(x) :
    e = np.exp(1)
    return 1/(1 + e**(-x))
def ReLu(x) :
    if x<=0 :
        return 0
    else :
        return x
def identity(x):
    return x
def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x-c)
    sum = np.sum(exp_a)
    return exp_a/sum

## 3개층 신경망. 

X = np.array([1.0 , 0.5, 1.0])
W1 = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6],[0.1,0.2,0.3]])
A1 = np.dot(X, W1)
Z1 = sigmoid(A1) #1행 3열 
print(Z1)
Z1 = np.append(Z1, 1.0)
W2 = np.array([[0.1,0.4], [0.2, 0.5], [0.3, 0.6], [0.1, 0.2]])
A2 = np.dot(Z1, W2)
Z2 = sigmoid(A2)
print(Z2)
Z2 =np.append(Z2, 1.0)
W3 = np.array([[0.1, 0.3],[0.2, 0.4],[0.1, 0.2]])
A3 = np.dot(Z2, W3)
Y = identity(A3)
print(Y)
Y_soft = softmax(A3)
print(Y_soft)