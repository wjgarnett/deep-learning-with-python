# !/usr/bin/env python
# coding: utf-8
'''
使用预训练的网络模型解决小数据集问题
'''
import os
import numpy as np
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers

# 载入VGG网络卷积层
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.summary()
batch_size = 20
datagen = ImageDataGenerator(rescale=1.0 / 255)

def extract_features(dir, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))    #conv_base最顶层的输出size
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(dir, target_size=(150, 150), batch_size=batch_size, class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i+1)*batch_size] = features_batch
        labels[i * batch_size : (i+1)*batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
    return model

if __name__ == '__main__':
    #数据路径
    base_dri = r'./dataset/dogs-vs-cats-small'
    train_dir = os.path.join(base_dri, 'train')
    validation_dir = os.path.join(base_dri, 'validation')
    test_dir = os.path.join(base_dri, 'test')

    train_features, train_labels = extract_features(train_dir, 2000)
    validation_features, validation_labels = extract_features(validation_dir, 1000)
    test_features, test_labels = extract_features(test_dir, 1000)

    train_features = np.reshape(train_features, (2000, 4*4*512))
    validation_features = np.reshape(validation_features, (1000, 4*4*512))
    test_features = np.reshape(test_features, (1000, 4*4*512))

    model = build_model()
    histroy = model.fit(train_features, train_labels, epochs=30, batch_size=20, validation_data=(validation_features, validation_labels), verbose=2)



