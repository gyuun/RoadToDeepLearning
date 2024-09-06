"""Handmade padding process module using list not ndarray"""
import numpy as np

def padding_2d(matrix : list, pad : int) -> list:
    """2dim array padding function"""
    padded_matrix = []
    row = len(matrix)
    col = len(matrix[0])
    padded_row = row + 2*pad
    padded_col = col + 2*pad
    for i in range(padded_row):
        new_col = [0 for i in range(padded_col)]
        if (i >= pad) and (i < pad + row):
            padded_matrix.append(new_col[:pad] + matrix[i-pad] + new_col[:pad])
        else:
            padded_matrix.append(new_col)
    return padded_matrix

def padding(tensor : np.array, pad : int) -> np.array:
    """4dim array pading function"""
    n, c = tensor.shape[:2]
    padded_tensor = []
    for i in range(n):
        new_channel = []
        for j in range(c):
            print(list(tensor[i][j]))
            new_matrix = padding_2d(tensor[i][j].tolist(), pad)
            new_channel.append(new_matrix)
        padded_tensor.append(new_channel)

    result = np.array(padded_tensor)

    return result
