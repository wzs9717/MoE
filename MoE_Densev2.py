# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 19:45:26 2019

@author: wzs
"""
import h5py
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from DenseMoE import DenseMoE
from scipy.io import savemat
import scipy.io as scio
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,TensorBoard
import datetime
from keras.models import Sequential,load_model
def train_main():
    n_hid_dim = 10
    # which_model='dense' # 'dense' or 'moe'
    batch_size=4
    epochs=20
    ##read files
    train_x = scio.loadmat(r'C:\Users\wzs\Desktop\DFCN\data_feature_train.mat')['train']
    test_x = scio.loadmat(r'C:\Users\wzs\Desktop\DFCN\data_feature_test.mat')['test']
    f = h5py.File(r'C:\Users\wzs\Desktop\DFCN\train_dataset.h5','r')                             
    train_y_orig = f['train_labels'][:] 
    train_y=to_categorical(train_y_orig) 
    f = h5py.File(r'C:\Users\wzs\Desktop\DFCN\test_dataset.h5','r')                            
    test_y_orig = f['test_labels'][:] 
    test_y=to_categorical(test_y_orig)
    ## make model
    input_shape = train_x.shape[1:]
    inputs = Input(shape=input_shape)
    

    hidden = Dense(n_hid_dim, activation='softmax',kernel_initializer='RandomUniform')(inputs)
    
    model = Model(inputs=inputs, outputs=hidden)
    print("Model created")
    model.summary()
#    log_dir=".\logs\fit\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    tbCallBack = TensorBoard(log_dir='.\logs\moe_2', 
                 histogram_freq=0,
#                  batch_size=32,     
                 write_graph=False,  
                 write_grads=False, 
                 write_images=False,
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
    ##train model
    opt = keras.optimizers.SGD(lr=1e-5, decay=1e-7)
    model.load_weights("./weights/main.h5")
    print("Model loaded.")

    lr_reducer= ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                      cooldown=0, patience=10, min_lr=0.5e-6)
    early_stopper= EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
    model_checkpoint= ModelCheckpoint("weights/main.h5", monitor="val_acc", save_best_only=True,
                                    save_weights_only=False)
    callbacks=[lr_reducer, early_stopper, model_checkpoint, tbCallBack]

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    hist = model.fit(train_x[:,:], train_y[:,:],
          batch_size=batch_size,
          callbacks=callbacks,
          epochs=epochs,
          validation_data=(test_x[0:2000,:], test_y[0:2000,:]),
          shuffle=True)
    ## Score trained model.
    scores = model.evaluate(test_x, test_y, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
   
    
def main_eval():
    #eval the main model to get num_catch
    model=load_model(r'./weights/main.h5')

    data = scio.loadmat(r'C:\Users\wzs\Desktop\DFCN\data_feature_train.mat')['train']
    # data.shape=[data.shape[0],28,28]
 
# feature extraction
    layer_1 = K.function([model.input], [model.output])
    num_catch=np.zeros(([data.shape[0],10]))
    for i in range(data.shape[0]):
        
        f1 = layer_1([data[i,:].reshape(1,1024)])[0]
        if i%100==0:
            print(i)
        num_catch[i,:]=f1#record the num
    scio.savemat('./data/num_catch.mat', {'num_catch':num_catch})

def train_subs():
    selector = scio.loadmat(r'./data/selector.mat')['selector']
#load data
    f = h5py.File(r'C:\Users\wzs\Desktop\DFCN\train_dataset.h5','r')                             
    train_y_orig = f['train_labels'][:] 
    train_y=to_categorical(train_y_orig) 
#train sub models
    dic_sub ={}
    for i in range(selector.shape[0]):
        print(i)
        ind=selector@train_y.T
        ind_s=np.nonzero(ind!=0)#get ind_s
        
        sub_train(ind_s,i)
    

def sub_train(ind_s,i):
#prepare inputs
    train_x = scio.loadmat(r'C:\Users\wzs\Desktop\DFCN\data_feature_train.mat')['train']
    test_x = scio.loadmat(r'C:\Users\wzs\Desktop\DFCN\data_feature_test.mat')['test']
    f = h5py.File(r'C:\Users\wzs\Desktop\DFCN\train_dataset.h5','r')                             
    train_y_orig = f['train_labels'][:] 
    train_y=to_categorical(train_y_orig) 
    f = h5py.File(r'C:\Users\wzs\Desktop\DFCN\test_dataset.h5','r')                            
    test_y_orig = f['test_labels'][:] 
    test_y=to_categorical(test_y_orig)
     ## make model
    input_shape = train_x.shape[1:]
    inputs = Input(shape=input_shape)
    
    train_x=train_x[ind_s,:]
    test_x=test_x[ind_s,:]
    train_y=train_y[ind_s,:]
    test_y=test_y[ind_s,:]
    
    n_hidden=100
    n_out = 10
    hidden1 = Dense(n_hidden, activation='relu',kernel_initializer='RandomUniform')(inputs)
    hidden2=Dense(n_out, activation='softmax',kernel_initializer='RandomUniform')(hidden1)
    model = Model(inputs=inputs, outputs=hidden2)
    print("Model created")
    model.summary()
#    log_dir=".\logs\fit\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    tbCallBack = TensorBoard(log_dir='.\logs\moe_2', 
                 histogram_freq=0,
#                  batch_size=32,     
                 write_graph=False,  
                 write_grads=False, 
                 write_images=False,
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
    ##train model
    opt = keras.optimizers.SGD(lr=1e-5, decay=1e-7)
    model.load_weights("./weights/main.h5")
    print("Model loaded.")

    lr_reducer= ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                      cooldown=0, patience=10, min_lr=0.5e-6)
    early_stopper= EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
    model_checkpoint= ModelCheckpoint("weights/main.h5", monitor="val_acc", save_best_only=True,
                                    save_weights_only=False)
    callbacks=[lr_reducer, early_stopper, model_checkpoint, tbCallBack]

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    hist = model.fit(train_x[:,:], train_y[:,:],
          batch_size=batch_size,
          callbacks=callbacks,
          epochs=epochs,
          validation_data=(test_x[0:2000,:], test_y[0:2000,:]),
          shuffle=True)
    ## Score trained model.
    scores = model.evaluate(test_x, test_y, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
def main():
    # train_main()
    # main_eval()
    train_subs()
    
    
if __name__ == '__main__':
    main()

#之后需要改一改sub的输出曾，因为不需要10个全用，只用2个就行