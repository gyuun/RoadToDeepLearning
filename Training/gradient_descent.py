import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from numerical_diff import function_2, gradient

def gradient_descent(f, init_x, lr= 0.01, step_num = 100):
    x= init_x

    for i in range(step_num):
        grad = gradient(f, x)
        x -= lr* grad
        print(x)
    return x

gradient_descent(function_2, np.array([-3.0, 4.0]), lr= 0.1, step_num= 100)
