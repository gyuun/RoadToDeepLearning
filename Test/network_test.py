"""Testing acivation functions"""
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
from ch02_Network import network

def test_stair():
    """계단 함수 테스트"""
    assert network.stair(3) == 1

def test_relu():
    """렐루 함수 테스트"""
    assert network.relu(3) == 3
