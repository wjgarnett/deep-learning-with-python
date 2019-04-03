# !/usr/bin/env python
# coding: utf-8
'''
目的： 熟悉CNN基本操作
'''

from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np
from keras.utils import to_categorical

def load_mnist_dataset():
    print("loading mnist dataset")
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    print("train_data.shape: ", train_data.shape)
    print("test_data.shape: ", test_data.shape)
    return (train_data, train_labels), (test_data, test_labels)

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = load_mnist_dataset()
    print(type(train_data))

    #准备数据
    train_images = train_data.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255.0
    test_images = test_data.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255.0
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = build_model()
    model.fit(train_images, train_labels, epochs=20, batch_size=64)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("test_loss: ", test_loss)
    print("test_acc: ", test_acc)


