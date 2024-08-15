"""Testing PerceptronLogic Class"""
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
from ch01_Perceptron import perceptron

class TestClass:
    """테스트 클래스에는 test prefix 가 존재해야함"""
    def __init__(self, x):
        """x = 1dim ndarray"""
        self.p = perceptron.PerceptronLogic(x)

    def test_and(self):
        """테스트 함수에는 test_ prefix 가 존재해야함 """
        assert self.p.and_gate() == 0

    def test_or(self):
        """or게이트 테스트"""
        assert self.p.and_gate() == 0
