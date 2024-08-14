"""Testing Layer function"""
import os
import sys
sys.path.append(os.getcwd()) # pylint: disable=wrong-import-position
from function_layer import MulLayer, AddLayer

if __name__ == '__main__':
    APPLE = 100
    APPLE_NUM = 2
    ORANGE = 150
    ORANGE_NUM = 3
    TAX= 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_fruit_layer = AddLayer()
    mul_tax_layer = MulLayer()

    APPLE_PRICE = mul_apple_layer.forward(APPLE, APPLE_NUM)
    ORANGE_PRICE = mul_orange_layer.forward(ORANGE, ORANGE_NUM)
    FRUIT_PRICE = add_fruit_layer.forward(APPLE_PRICE, ORANGE_PRICE)
    price = mul_tax_layer.forward(FRUIT_PRICE, TAX)

    DPRICE = 1
    dall_price , dtax = mul_tax_layer.backward(DPRICE)
    dapple_price, dorange_price = add_fruit_layer.backward(dall_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(dapple_num, dapple, dorange, dorange_num, dtax)
