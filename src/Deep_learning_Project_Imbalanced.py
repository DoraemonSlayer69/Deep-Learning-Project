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
from sklearn.metrics import confusion_matrix
from keras.applications import VGG16
from sklearn.metrics import classification_report
from keras.applications import Xception
from keras.models import load_model

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

def ReportGen(y_pred,true_labels):
    true_labels = np.array(true_labels)
    y_pred = (y_pred > 0.5)
    true_labels = (true_labels > 0.5)
    conf_matrix = confusion_matrix(true_labels,y_pred)
    print(conf_matrix)
    print('Report : ')
    print(classification_report(true_labels,y_pred))



base_dir = 'D:\Personal\Deep Learning Project\chest_xray'

custom_dir = 'D:\Personal\Deep Learning Project\Chest_Xrays_ibalanced'
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
test_dir_normal = os.path.join(test_dir,'NORMAL')
test_dir_pneumonia = os.path.join(test_dir,'PNEUMONIA')
#Creating Train and Test and val directories
if os.path.exists(custom_dir)==False:
    os.mkdir(custom_dir)
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    os.mkdir(val_dir)
    os.mkdir(train_dir_normal)
    os.mkdir(val_dir_normal)
    os.mkdir(train_dir_pneumonia)
    os.mkdir(val_dir_pneumonia)
    os.mkdir(test_dir_normal)
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
    if i < 3000:
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
    

for i in range(3000,len(filenames_pneumonia)):
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
        batch_size=24,
        class_mode='binary')


    
#Desiging the network
def Custom_neuralNetwork():
    
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
    network.compile(optimizer=optimizers.Adam(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])
    network.summary()
    return network



custom_network = Custom_neuralNetwork()
history = custom_network.fit_generator(train_gen,steps_per_epoch=100,epochs=40,validation_data=val_gen,validation_steps=100)
custom_network.save("./models/ChestXraysCustomCNNmodel")

PlotGraph(history)

results = []
for i in range(26):
    x_test,y_test = next(test_gen)
    results.append(y_test)

  #true_labels = list(itertools.chain(results[i]))    
import itertools
true_labels=list(itertools.chain.from_iterable(results))

network = load_model("./models/ChestXraysCustomCNNmodel")
y_pred = custom_network.predict_generator(test_gen,verbose=1)
ReportGen(y_pred,true_labels)


conv_base = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape=(224,224,3))
conv_base.summary()


def VGG16NetworkFrozen():
#VGG16Model 
    conv_base.trainable = False
    network = models.Sequential()
    network.add(conv_base)
    network.add(layers.Flatten())
    network.add(layers.Dense(units=256, activation='relu'))
    network.add(layers.Dropout(0.5))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(1,activation='sigmoid'))
    network.summary()
    network.compile(optimizer=optimizers.Adam(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])

VGG16Network = VGG16NetworkFrozen()
history = VGG16Network.fit_generator(train_gen,steps_per_epoch=100,epochs=40,validation_data=val_gen,validation_steps=100)
VGG16Network.save("VGG16ChestXraysFrozen.h5")

PlotGraph(history)
def VGG16finetuned():
    #Training again by unfreezing some layers

    conv_base.trainable = True
    set_trainable = False

    for layer in conv_base.layers:
        if layer.name == 'block4_conv2':
            set_trainable=True
        if set_trainable:
            layer.trainable=True
        else:
            layer.trainable=False

    conv_base.summary()  
    network = models.Sequential()
    network.add(conv_base)
    network.add(layers.Flatten())
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(units=512, activation='relu'))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(units=512, activation='relu'))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(1,activation='sigmoid'))
    network.summary()
    network.compile(optimizer=optimizers.Adam(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])
    return network



UnfrozenVGG16 = VGG16finetuned()

history = UnfrozenVGG16.fit_generator(train_gen,steps_per_epoch=100,epochs=40,validation_data=val_gen,validation_steps=100)
PlotGraph(history)

results = []
for i in range(26):
    x_test,y_test = next(test_gen)
    results.append(y_test)



    #true_labels = list(itertools.chain(results[i]))    
import itertools
true_labels=list(itertools.chain.from_iterable(results))


    

#For VGG16Unfrozen
network = models.load_model("VGG16ChestXraysUnFrozen.h5")
network.summary()
y_pred = network.predict_generator(test_gen,verbose=1)
ReportGen(y_pred,true_labels)

#Changing resolution of Images for Xception
train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(299,299),
        batch_size=20,
        class_mode='binary')

val_gen = test_datagen.flow_from_directory(
        val_dir,
        target_size=(299,299),
        batch_size=20,
        class_mode='binary')


test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(299,299),
        batch_size=24,
        class_mode='binary')

#Xception model
conv_xception = Xception(include_top=False, weights='imagenet', input_shape = (299, 299, 3))
conv_xception.summary()
def XceptionModel():
    conv_xception.trainable = False
    # Model create
    model = models.Sequential()
    model.add(conv_xception)
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    print(model.summary())
    #model compile
    opt = optimizers.Adam(lr = 1e-4)
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['acc'])
    return model

#Fit model
model = XceptionModel()
history = model.fit_generator(train_gen, steps_per_epoch = 100, epochs = 40,
                              validation_data = val_gen, validation_steps = 100)

model.save("Xception.h5")
accuracy = model.evaluate(test_gen, verbose = 1)

def Xceptionfinetuned():
    #Training again by unfreezing some layers

    conv_xception.trainable = True
    set_trainable = False

    for layer in conv_xception.layers:
        if layer.name == 'block11_sepconv2':
            set_trainable=True
        if set_trainable:
            layer.trainable=True
        else:
            layer.trainable=False

    conv_xception.summary()  
    network = models.Sequential()
    network.add(conv_xception)
    network.add(layers.Flatten())
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(units=1024, activation='relu'))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(units=512, activation='relu'))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(1,activation='sigmoid'))
    network.summary()
    network.compile(optimizer=optimizers.Adam(lr=1e-5),loss='binary_crossentropy',metrics=['acc'])
    return network

network = Xceptionfinetuned()

history = model.fit_generator(train_gen, steps_per_epoch = 100, epochs = 40,
                              validation_data = val_gen, validation_steps = 100)
PlotGraph(history)
network.save("Xception_finetuned.h5")
loss,accuracy = network.evaluate_generator(test_gen,verbose=1)

results = []
for i in range(26):
    x_test,y_test = next(test_gen)
    results.append(y_test)
    
import itertools
true_labels=list(itertools.chain.from_iterable(results))

network = load_model('Xception.h5')
y_pred = network.predict_generator(test_gen,verbose=1)
ReportGen(y_pred,true_labels)
