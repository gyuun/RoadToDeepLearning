"""Providing perceptron class"""
import numpy as np

class PerceptronLogic :
    """perceptron logic"""
    def __init__ (self, x):
        """x : ndarray"""
        self.x = np.append(x, 1)

    def set_attribute(self, x):
        """ set x for new one """
        self.x = np.append(x, 1)

    def result(self, w):
        """ Affine 결과 """
        if np.sum(np.dot(self.x, w)) > 0:
            return 1
        return 0

    def and_gate(self):
        """ and 게이트 """
        w = np.array([0.5, 0.5, -0.7])
        return self.result(w)

    def nand_gate(self):
        """ nand 게이트 """
        w = np.array([-0.5, -0.5, 0.7])
        return self.result(w)

    def or_gate(self):
        """ or 게이트 """
        w = np.array([0.5, 0.5, -0.3])
        return self.result(w)

    def xor_gate(self):
        """ xor 게이트 """
        s1 = self.nand_gate()
        s2 = self.or_gate()
        new_neuron = PerceptronLogic(np.array([s1, s2]))

        return new_neuron.and_gate()
