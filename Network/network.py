"""Providing activation functions"""
import numpy as np

def stair(x):
    """0,1의 계단함수"""
    if x<=0 :
        return 0
    return 1

def sigmoid(x) :
    """sigmoid 함수"""
    e = np.exp(1)
    return 1/(1 + e**(-x))

def relu(x) :
    """ReLu 함수"""
    return np.maximum(0,x)

def identity(x):
    """항등 함수"""
    return x

def softmax(x):
    """소프트맥스 함수"""
    c = np.max(x)
    exp_a = np.exp(x-c)
    other_sum = np.sum(exp_a)
    return exp_a/other_sum
