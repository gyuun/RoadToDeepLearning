"""Providing im2col function"""
import numpy as np

def im2col(image:np.array , fw: int, fh: int, stride: int, pad: int) -> np.array:
    """image : 4dim (N, C, W, H) -> 2dim"""
    #패딩처리
    image = np.pad(image,
                   pad_width=((0,0),(0,0),(pad,pad),(pad,pad)),
                   mode='constant', constant_values=0)
    n, c, w, h = image.shape
    oh = int((h - fh) / stride) + 1 # oh
    ow = int((w - fw) / stride) + 1 # ow
    #변환
    result = np.zeros((oh*ow*n,fw*fh*c))
    for num in range(n):
        for patch in range(oh*ow): # oh*ow = num of patch
            patch_y = patch // oh * stride
            patch_x = patch % ow * stride

            result[patch + num*oh*ow, :] = \
            image[num, :, patch_y:patch_y+fh, patch_x:patch_x+fw].reshape(1, fh*fw*c)

    return result
