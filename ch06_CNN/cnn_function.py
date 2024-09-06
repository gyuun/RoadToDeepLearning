"""Providing im2col function"""
import numpy as np

def im2col(image:np.array , fw: int, fh: int, stride: int, pad: int) -> np.array:
    """image : 4dim (N, C, W, H) -> 2dim"""
    #패딩처리
    image = np.pad(image,
                   pad_width=((0,0),(0,0),(pad,pad),(pad,pad)),
                   mode='constant', constant_values=0)
    shape_list = list(image.shape)
    out_layer = []
    out_layer.append(int((shape_list[2] - fh) / stride) + 1) # h
    out_layer.append(int((shape_list[3] - fw) / stride) + 1) # w
    #변환
    num_of_patch = out_layer[0] * out_layer[1]
    result = np.array([])
    for num in range(shape_list[0]):
        for patch in range(num_of_patch):
            patch_y = patch // out_layer[0] * stride
            patch_x = patch % out_layer[1] * stride
            for channel in range(shape_list[1]):
                if patch_x == 0:
                    column = image[num, channel, patch_y:patch_y+fh, patch_x:fw]
                else:
                    column = image[num, channel, patch_y:patch_y+fh, patch_x:patch_x+fw]
                column = column.reshape(1,fh*fw) # 2dim
                result = np.append(result, column)

    result = result.reshape((num_of_patch*shape_list[0],fw*fh*shape_list[1]))

    return result
