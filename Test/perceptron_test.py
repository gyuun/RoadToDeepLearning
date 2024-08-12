"""Testing PerceptronLogic Class"""
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from Perceptron import perceptron

class TestClass:
    """테스트 클래스에는 test prefix 가 존재해야함"""
    def test_and(self):
        """테스트 함수에는 test_ prefix 가 존재해야함 """
        x = np.array([0,0])
        p = perceptron.perceptronLogic(x)
        assert p.AND() == 0
