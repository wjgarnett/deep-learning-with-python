# !/usr/bin/env python  #首行目的：告诉操作系统用什么方式执行该脚本
# coding: utf-8

from keras.datasets import boston_housing
from keras import layers
from keras import models
import numpy as np

def load_boston_housing_dataset():
    print("loading boston housing dataset")
    (train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
    print("train_data.shape: ", train_data.shape)
    print("test_data.shape: ", test_data.shape)
    return (train_data, train_labels), (test_data, test_labels)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1)) #回归问题最后一层通常没有激活函数
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae']) #mse: mean squared error; mae: mean absolute error
    return model

def k_fold_validation(train_data, train_targets, k):
    num_epoch = 200
    num_val_sample = len(train_data) // k;
    all_scores = []
    for i in range(k):
        print("-----processing fold #", i)
        #划分测试集&验证集
        val_data = train_data[i*num_val_sample: (i+1)*num_val_sample]
        val_targets = train_targets[i*num_val_sample: (i+1)*num_val_sample]
        partial_train_data = np.concatenate([train_data[0 : i*num_val_sample], train_data[(i+1)*num_val_sample : ]], axis=0)
        partial_train_targets = np.concatenate([train_targets[0 : i*num_val_sample], train_targets[(i+1)*num_val_sample : ]], axis=0)
        # print("val_data.shape: ", val_data.shape)
        # print("val_targets.shape: ", val_targets.shape)
        # print("partial_train_data.shape: ", partial_train_data.shape)
        # print("partial_train_targets.shape: ", partial_train_targets.shape)
        model = build_model()
        # verbose：日志显示 0-不显示日志 1-输出进度条记录verbe 2-为每个epoch输出一行记录
        model.fit(partial_train_data, partial_train_targets, epochs=num_epoch, batch_size=1, verbose=1)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
        all_scores.append(val_mae)
    return all_scores





if __name__ == '__main__':
    # load dataset
    (train_data, train_targets), (test_data, test_targets) = load_boston_housing_dataset()
    print(type(train_data))

    #prepare the data
    mean = train_data.mean(axis=0)  #axis=0,即把该维度压缩为1
    train_data -= mean
    std = train_data.var(axis=0)
    train_data /= std

    test_data -= mean   #实际中，测试集的均值方差没法计算，用训练集的统计值来标准化测试集数据
    test_data /= std

    all_scores = k_fold_validation(train_data, train_targets, 4)
    print("all_scores: ", np.mean(all_scores))