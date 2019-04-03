# !/usr/bin/env python
# coding: utf-8
'''
使用预训练的网络模型解决小数据集问题
'''

from keras.applications import VGG16

if __name__ == '__main__':
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape = (150, 150, 3))
    conv_base.summary()