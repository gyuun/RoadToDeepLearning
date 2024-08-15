"""Testing loss functions"""
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
import numpy as np
from ch03_Training import loss_function

def test_square():
    """오차제곱합 테스트"""
    t = np.array([0,0,1,0,0,0,0,0,0,0])
    y = np.array([0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0])
    assert loss_function.sum_square_error(y, t)

def test_cross_entropy():
    """교차엔트로피합 테스트"""
    t = np.array([0,0,1,0,0,0,0,0,0,0])
    y = np.array([0.1, 0.05, 0.6, 0, 0.05, 0.1, 0, 0.1, 0, 0])
    assert loss_function.cross_entropy_error(y, t)
