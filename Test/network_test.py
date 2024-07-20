import pytest
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Network import network

def test_stair():
    assert network.stair(3) == 1
def test_ReLu():
    assert network.ReLu(3) == 3
