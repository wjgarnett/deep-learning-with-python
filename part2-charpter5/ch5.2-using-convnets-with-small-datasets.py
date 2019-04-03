# !/usr/bin/env python
# coding: utf-8
'''
目的： 实际中遇到的通常都是在小批量数据集上进行深度学习算法研发的情况，本例子基于小批量数据集从头开始许训练一个CNN网络模型
'''
import os
import random
import shutil
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

g_original_dataset_dir = r'./dataset/kaggle/train'  # 原始数据集
g_dst_root = r'./dataset/dogs-vs-cats-small'  # 小批量数据集存放根目录

# 在kaggle dogs vs cats数据集上，创建小批量数据集
def create_small_dataset():
    global g_original_dataset_dir, g_base_dir

    #创建根目录
    if not os.path.exists(g_dst_root):
        os.mkdir(g_dst_root)
    #按数据集划分创建子目录
    train_dir = os.path.join(g_dst_root, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    validation_dir = os.path.join(g_dst_root, 'validation')
    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)
    test_dir = os.path.join(g_dst_root, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    #创建类别子目录
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    if not os.path.exists(train_dogs_dir):
        os.mkdir(train_dogs_dir)
    train_cats_dir = os.path.join(train_dir, 'cats')
    if not os.path.exists(train_cats_dir):
        os.mkdir(train_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    if not os.path.exists(validation_dogs_dir):
        os.mkdir(validation_dogs_dir)
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    if not os.path.exists(validation_cats_dir):
        os.mkdir(validation_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    if not os.path.exists(test_dogs_dir):
        os.mkdir(test_dogs_dir)
    test_cats_dir = os.path.join(test_dir, 'cats')
    if not os.path.exists(test_cats_dir):
        os.mkdir(test_cats_dir)

    #从原始数据中拷贝数据到小批量数据集中
    dogs_file_list = []
    cats_file_list = []
    for file in os.listdir(g_original_dataset_dir):
        if file[0:3] == 'cat':
            cats_file_list.append(file)
        elif file[0:3] == 'dog':
            dogs_file_list.append(file)
    random.shuffle(dogs_file_list)
    random.shuffle(cats_file_list)
    for file in dogs_file_list[0:1000]:
        shutil.copy(os.path.join(g_original_dataset_dir, file), os.path.join(train_dogs_dir, file))
    for file in dogs_file_list[1000:1500]:
        shutil.copy(os.path.join(g_original_dataset_dir, file), os.path.join(validation_dogs_dir, file))
    for file in dogs_file_list[1500:2000]:
        shutil.copy(os.path.join(g_original_dataset_dir, file), os.path.join(test_dogs_dir, file))
    for file in cats_file_list[0:1000]:
        shutil.copy(os.path.join(g_original_dataset_dir, file), os.path.join(train_cats_dir, file))
    for file in cats_file_list[1000:1500]:
        shutil.copy(os.path.join(g_original_dataset_dir, file), os.path.join(validation_cats_dir, file))
    for file in cats_file_list[1500:2000]:
        shutil.copy(os.path.join(g_original_dataset_dir, file), os.path.join(test_cats_dir, file))

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))  #添加dropout层抑制过拟合
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['acc'])
    return model

def save_figures(history, contents):
    for content in contents:
        train_content = history.history[content[0]]
        validation_content = history.history[content[1]]
        epochs = range(1, len(train_content) + 1)
        plt.plot(epochs, train_content, 'bo', label=content[0])
        plt.plot(epochs, validation_content, 'b', label=content[1])
        title = ("%s and %s" % (content[0], content[1]))
        plt.title(title)
        plt.legend()
        plt.savefig(title + ".jpg")
        plt.clf()

if __name__ == '__main__':
    # create_small_dataset()

    #data processing
    train_dir = r'./dataset/dogs-vs-cats-small/train'
    # train_datagen = ImageDataGenerator(rescale=1./255)
    # 数据增强
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                       shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
    validaiton_dir = r'./dataset/dogs-vs-cats-small/validation'
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_directory(validaiton_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

    #build model
    model = build_model()

    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)
    model.save("dogs-and-cats-small.h5")
    save_figures(history,[("acc", "val_acc"), ("loss", "val_loss")])


