# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:28:17 2020

@author: SHIRISH
"""
import os, shutil
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import regularizers
from keras import optimizers

base_dir = 'D:\Personal\Deep Learning Project\chest_xray'

custom_dir = 'D:\Personal\Deep Learning Project\Chest_Xrays'
original_train = os.path.join(base_dir,'train')
original_test = os.path.join(base_dir,'test')
original_train_normal = os.path.join(original_train,"NORMAL")
original_train_pneumonia = os.path.join(original_train,"PNEUMONIA")
original_test_normal = os.path.join(original_test,'NORMAL')
original_test_pneumonia = os.path.join(original_test,'PNEUMONIA')
#Creating Train and Test and val directories
if os.path.exists(custom_dir)==False:
    os.mkdir(custom_dir)
    train_dir = os.path.join(custom_dir,'train')
    os.mkdir(train_dir)
    test_dir = os.path.join(custom_dir,'test')
    os.mkdir(test_dir)
    val_dir = os.path.join(custom_dir,'val')
    os.mkdir(val_dir)
    train_dir_normal = os.path.join(train_dir,'NORMAL')
    os.mkdir(train_dir_normal)
    val_dir_normal = os.path.join(val_dir,'NORMAL')
    os.mkdir(val_dir_normal)
    train_dir_pneumonia = os.path.join(train_dir,'PNEUMONIA')
    os.mkdir(train_dir_pneumonia)
    val_dir_pneumonia = os.path.join(val_dir,'PNEUMONIA')
    os.mkdir(val_dir_pneumonia)
    test_dir_normal = os.path.join(test_dir,'NORMAL')
    os.mkdir(test_dir_normal)
    test_dir_pneumonia = os.path.join(test_dir,'PNEUMONIA')
    os.mkdir(test_dir_pneumonia)
    
    
filenames_pneumonia = os.listdir(original_train_pneumonia)
filenames_normal = os.listdir(original_train_normal)
test_filenames_normal = os.listdir(original_test_normal)
test_filenames_pneumonia = os.listdir(original_test_pneumonia)        
#Training Set
for i in range(0,len(filenames_normal)):
    if i < 1000:
        src = os.path.join(original_train_normal,filenames_normal[i])
        dst = os.path.join(train_dir_normal,filenames_normal[i])
        if os.path.exists(dst) == False:
            shutil.copy(src,dst)
    else:
        break

for i in range(0,len(filenames_pneumonia)):
    if i < 1000:
        src = os.path.join(original_train_pneumonia,filenames_pneumonia[i])
        dst = os.path.join(train_dir_pneumonia,filenames_pneumonia[i])
        
        shutil.copy(src,dst)
    else:
        break


#Validation set
for i in range(1000,len(filenames_normal)):
        src = os.path.join(original_train_normal,filenames_normal[i])
        dst = os.path.join(val_dir_normal,filenames_normal[i])
        if os.path.exists(dst) == False:
            shutil.copy(src,dst)
    

for i in range(1000,len(filenames_normal)):
        src = os.path.join(original_train_pneumonia,filenames_pneumonia[i])
        dst = os.path.join(val_dir_pneumonia,filenames_pneumonia[i])
        if os.path.exists(dst) == False:
            shutil.copy(src,dst)

#Test set
for i in range(0,len(test_filenames_normal)):
        src = os.path.join(original_test_normal,test_filenames_normal[i])
        dst = os.path.join(test_dir_normal,test_filenames_normal[i])
        if os.path.exists(dst) == False:
            shutil.copy(src,dst)

for i in range(0,len(test_filenames_pneumonia)):
        src = os.path.join(original_test_pneumonia,test_filenames_pneumonia[i])
        dst = os.path.join(test_dir_pneumonia,test_filenames_pneumonia[i])
        if os.path.exists(dst) == False:
            shutil.copy(src,dst)
    
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


test_datagen = ImageDataGenerator(
        rescale=1./255)


train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=20,
        class_mode='binary')

val_gen = test_datagen.flow_from_directory(
        val_dir,
        target_size=(224,224),
        batch_size=20,
        class_mode='binary')


#Desiging the network
network = models.Sequential()
network.add(layers.Conv2D(32, (3,3), activation='relu',kernel_regularizer=regularizers.l2(0.001),input_shape=(224,224,3)))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64, (3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(100, (3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(100, (3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(128, (3,3),kernel_regularizer=regularizers.l2(0.001),activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Flatten())
network.add(layers.Dropout(0.5))
network.add(layers.Dense(units=512, activation='relu'))
network.add(layers.Dense(1,activation='sigmoid'))
network.summary()


network.compile(optimizer=optimizers.Adam(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])

history = network.fit_generator(train_gen,steps_per_epoch=100,epochs=10,validation_data=val_gen,validation_steps=100)
