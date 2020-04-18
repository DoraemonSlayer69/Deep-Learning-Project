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
from keras.preprocessing import image
import numpy as np
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


base_dir = 'D:\Personal\Deep Learning Project\chest_xray'

custom_dir = 'D:\Personal\Deep Learning Project\Chest_Xrays'
original_train = os.path.join(base_dir,'train')
original_test = os.path.join(base_dir,'test')
original_train_normal = os.path.join(original_train,"NORMAL")
original_train_pneumonia = os.path.join(original_train,"PNEUMONIA")
original_test_normal = os.path.join(original_test,'NORMAL')
original_test_pneumonia = os.path.join(original_test,'PNEUMONIA')
train_dir = os.path.join(custom_dir,'train')
test_dir = os.path.join(custom_dir,'test')
val_dir = os.path.join(custom_dir,'val')
train_dir_normal = os.path.join(train_dir,'NORMAL')
train_dir_pneumonia = os.path.join(train_dir,'PNEUMONIA')
val_dir_normal = os.path.join(val_dir,'NORMAL')
val_dir_pneumonia = os.path.join(val_dir,'PNEUMONIA')
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
        brightness_range=[0,2],
        zca_epsilon=2.0,
        fill_mode='nearest',
        zca_whitening=False,
        zoom_range=0.2,
        rotation_range=40)




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


test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224,224),
        batch_size=20,
        class_mode='binary')

import matplotlib.pyplot as plt
#loading image 
img_name = "D:/Personal/Deep Learning Project/Chest_Xrays/train/NORMAL/IM-0133-0001.jpeg"
print(img_name)
img = image.load_img(img_name,target_size=(224,224))
img_tensor = image.img_to_array(img)


aug_img = (1.5 * img_tensor) + 2
aug_img /=255.0
img_tensor /= 255.0

k=0
for i in train_gen:
    if k == 10:
        break
    else:
        img = i[0]
        #img = img/255.0
        plt.imshow(img[0])
        k+=1
        print(k)

#img_tensor = np.expand_dims(img_tensor, axis=0)
print(img_tensor.shape)
plt.imshow(img_tensor)
plt.imshow(aug_img)

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

history = network.fit_generator(train_gen,steps_per_epoch=100,epochs=40,validation_data=val_gen,validation_steps=100)

def PlotGraph(history):
    import matplotlib.pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1,len(acc) + 1)

#Validation vs Training in accuracy
    plt.plot(epochs,acc,'bo',label='training accuracy')
    plt.plot(epochs,val_acc,'b',label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs,loss,'bo',label='training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.figure()
    plt.show()

PlotGraph(history)


