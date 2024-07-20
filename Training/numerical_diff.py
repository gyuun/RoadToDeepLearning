import numpy as np

def numerical_diff (f, x):
    """반올림오차 발생 예시"""
    h = 1e-50
    return f(x + h) - f(x) /h

def central_num_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_2(x):
    """x1^2 + x2^2 를 반환한다.
    
    parameter : ndarray
    return : float
    """
    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0*x0
def function_tmp2(x1):
    return x1*x1

def partial_diff(f1, f2, x):
    """x1^2 + x2^2 의 편미분을 출력한다.
    
    parameter : 
    (f1 : function_tmp1)
    (f2 : function_tmp2)
    (x : ndarray)
    return : float
    """
    dx0 = central_num_diff(f1, x[0])
    dx1 = central_num_diff(f2, x[1])
    print('dx0 ', dx0, 'dx1', dx1)


def numerical_gradient(f, x):
    """f에 대해 x 에서의 gradient 를 반환한다.
    
    parameter : (f : 미분할 함수), (x : ndarray)
    return : ndarray
    """
    h = 1e-4
    if x.ndim == 1:
        grad = np.zeros_like(x)
        for idx in range(x.size):
            tmp_val = x[idx]
            x[idx] = tmp_val + h
            fxh1 = f(x)

            x[idx] = tmp_val - h
            fxh2 = f(x)
            grad[idx] = (fxh1 - fxh2) / (2*h)
            x[idx] = tmp_val
        return grad
    elif x.ndim == 2:
        res = []
        for i in range(x.shape[0]):
            grad_i = np.zeros_like(x[i])
            for idx in range(x[i].size):
                tmp_val = x[i, idx]
                x[i, idx] = tmp_val + h
                fxh1 = f(x)
                x[i, idx] = tmp_val - h
                fxh2 = f(x)
                grad_i[idx] = (fxh1 - fxh2) / (2*h)
                print(grad_i)
                x[i, idx] = tmp_val
            res.append(grad_i)
        return res







