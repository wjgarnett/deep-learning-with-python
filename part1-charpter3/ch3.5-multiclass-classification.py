# !/usr/bin/env python  #首行目的：告诉操作系统用什么方式执行该脚本
# coding: utf-8

'''
分类问题：
    binary classification
    multicalss classification
    multilabel classification
'''

import numpy as np
from keras.datasets import reuters #加载数据集
from keras.utils.np_utils import to_categorical #one-hot 编码
from keras import models
from keras import layers
import matplotlib.pyplot as plt

def load_reuters_dataset():
    print("loading reuters dataset")
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    print("len(train_data): ", len(train_data))
    print("len(test_data): ", len(test_data))
    return (train_data, train_labels), (test_data, test_labels)

# 将整数序列编码为二进制矩阵--->向量化便于NN处理
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    #加载数据
    (train_data, train_labels), (test_data, test_labels) = load_reuters_dataset()

    #数据预处理
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    x_val = x_train[: 1000]
    partial_x_train = x_train[1000: ]
    y_val = one_hot_train_labels[: 1000]
    partial_y_train = one_hot_train_labels[1000: ]

    model = build_model()
    history = model.fit(partial_x_train, partial_y_train, epochs=8, batch_size=512, validation_data=(x_val, y_val))

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'bo', label='training loss')
    # plt.plot(epochs, val_loss, 'b', label='validation loss')
    # plt.title('training and validation loss')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()
    #
    # plt.clf()
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # plt.plot(epochs, acc, 'bo', label='training acc')
    # plt.plot(epochs, val_loss, 'b', label='validation acc')
    # plt.title('training and validation accuracy')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()

    result = model.evaluate(x_test, one_hot_test_labels)
    print("result: ", result) #loss & accuracy

    predictions = model.predict(x_test)
    np.argmax(predictions[0])

    import copy
    # baseline的计算方式，测试集是非平衡的用这种方式计算baseline才对
    test_labels_copy = copy.copy(test_labels)
    np.random.shuffle(test_labels_copy)
    baseline = float( np.sum(np.array(test_labels) == np.array(test_labels_copy)) )/len(test_labels)
    print("baseline: ", baseline)




