"""Providing batch nomalization"""
import numpy as np
def batch_normalization(batch):
    """batch : 2차원 리스트"""
    normalized_batch = np.array([])

    for li in batch:
        batch_mean = np.sum(li) / len(li)
        batch_variation = np.sum((li - batch_mean)**2)/len(li)
        batch_sigma = np.sqrt(batch_variation)
        normalized = (li - batch_mean) / batch_sigma

        np.append(normalized_batch, normalized)

    return normalized_batch
