# !/usr/bin/env python  #首行目的：告诉操作系统用什么方式执行该脚本
# coding: utf-8

import numpy as np
from keras.datasets import imdb
from keras import models, layers, optimizers, losses, metrics
import matplotlib.pyplot as plt

# 将整数序列编码为二进制矩阵--->向量化便于NN处理
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])
    return model

if __name__ == '__main__':
    #加载IMDB dataset
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
    print(len(train_data))

    #数据编码为定长的，便于后续网络处理
    x_train = vectorize_sequences(train_data)
    y_train = np.asarray(train_labels).astype("float32")
    x_test = vectorize_sequences(test_data)
    y_test = np.asarray(test_labels).astype("float32")

    #将训练集拆分成两部分（数据少时需要用交叉验证，此处不需要。）
    #Q:数据多少如何定性判定？？？
    #验证集的目的是为了便于监督训练过程，合理配置超参等。
    x_val = x_train[: 10000]
    y_val = y_train[: 10000]
    patrial_x_train = x_train[10000 : ]
    patrial_y_train = y_train[10000 : ]

    model = build_model()
    history = model.fit(patrial_x_train, patrial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

    history_dict = history.history
    print(history_dict.keys())

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, loss, 'bo', label="training loss")
    plt.plot(epochs, val_loss, 'b', label="validation loss")
    plt.title('training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    plt.clf()
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc, 'ro', label='training acc')
    plt.plot(epochs, val_acc, 'r', label='validation acc')
    plt.title('training and validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    # 准确率评估（在测试集上评估准确率，验证集上监督训练过程便于调参等）
    results = model.evaluate(x_test, y_test)
    print(results)

    #
    results = model.predict(x_test)
    print(results)
