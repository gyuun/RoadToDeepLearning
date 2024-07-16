import pytest
import numpy as np
from perceptron import perceptronLogic

class TestClass:
    #테스트 클래스에는 test prefix 가 존재해야함
    def test_and(self):
        # 테스트 함수에는 test_ prefix 가 존재해야함
        x = np.array([0,0])
        p = perceptronLogic(x)
        assert p.AND() == 0
        