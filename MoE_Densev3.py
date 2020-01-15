#!/usr/bin/env python
# encoding: utf-8
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import io
import itertools
from packaging import version
from six.moves import range
import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import scipy.io as scio
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping,TensorBoard
import matplotlib.pyplot as plt
import pickle
import sklearn.metrics
mnist = tf.keras.datasets.mnist
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."
    
def tinyCNN():
    import tensorflow.keras.layers as K
    classes=2
    chanDim=-1#TF BK
    inputShape=(28, 28, 1)
    model = tf.keras.Sequential()
    model.add(K.Conv2D(32, (3, 3), padding="same",
    			input_shape=inputShape))
    model.add(K.Activation("relu"))
    model.add(K.BatchNormalization(axis=chanDim))
    model.add(K.Conv2D(32, (3, 3), padding="same"))
    model.add(K.Activation("relu"))
    model.add(K.BatchNormalization(axis=chanDim))
    model.add(K.MaxPooling2D(pool_size=(2, 2)))
    model.add(K.Dropout(0.25))
    
    # second CONV => RELU => CONV => RELU => POOL layer set
    model.add(K.Conv2D(64, (3, 3), padding="same"))
    model.add(K.Activation("relu"))
    model.add(K.BatchNormalization(axis=chanDim))
    model.add(K.Conv2D(64, (3, 3), padding="same"))
    model.add(K.Activation("relu"))
    model.add(K.BatchNormalization(axis=chanDim))
    model.add(K.MaxPooling2D(pool_size=(2, 2)))
    model.add(K.Dropout(0.25))
    
    # first (and only) set of FC => RELU layers
    model.add(K.Flatten())
    model.add(K.Dense(512))
    model.add(K.Activation("relu"))
    model.add(K.BatchNormalization())
    model.add(K.Dropout(0.5))
    
    # softmax classifier
    model.add(K.Dense(classes,name="new_Dense"))
    model.add(K.Activation("softmax",name="new_softmax"))
    return model
def train_subs():
    
    num_exp=5
    num_class=2#number of class that hard to classif
    cm = scio.loadmat(r'./data/cm.mat')['cm']
    for i in range(10):
        cm[i,i]=0
        
    selector=np.zeros(([10,num_exp]))
    for i in range(num_exp):
        ind=np.where(cm==np.max(cm))
        indx=ind[0][0]
        indy=ind[1][0]
        selector[[indx,indy],i]=1
        cm[indx,indy]=0
    selector=selector.T
    print(selector)
#load data
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = \
        fashion_mnist.load_data()
    test_y=to_categorical(test_labels)
    train_y=to_categorical(train_labels)
    
#train sub models
    dic_sub ={}
    for i in range(1):#change selector.shape[0]
        print(i)
        ind_train=selector[i,:]@train_y.T
        ind_test=selector[i,:]@test_y.T
        ind_s=[np.nonzero(ind_train!=0),np.nonzero(ind_test!=0)]#get ind_s
        
        sub_train(selector[i,:],ind_s,i,num_exp,num_class)
    scio.savemat('./data/selector.mat', {'selector':selector})

def sub_train(selector,ind_s,ind_exp,num_exp,num_class):
#prepare inputs
    
    ind_train=ind_s[0]
    ind_test=ind_s[1]
    # train_x = scio.loadmat(r'C:\Users\wzs\Desktop\DFCN\data_feature_train.mat')['train']
    # test_x = scio.loadmat(r'C:\Users\wzs\Desktop\DFCN\data_feature_test.mat')['test']
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = \
        fashion_mnist.load_data()
    train_x = train_images.reshape([-1,28,28,1]) / 255.0
    test_x = test_images.reshape([-1,28,28,1]) / 255.0
    # f = h5py.File(r'C:\Users\wzs\Desktop\DFCN\train_dataset.h5','r')                             
    # train_y_orig = f['train_labels'][:] 
    # train_y=to_categorical(train_y_orig) 
    # f = h5py.File(r'C:\Users\wzs\Desktop\DFCN\test_dataset.h5','r')                            
    # test_y_orig = f['test_labels'][:] 
    # test_y=to_categorical(test_y_orig)
    test_y=test_labels
    train_y=train_labels
    test_y=to_categorical(test_y)
    train_y=to_categorical(train_y)
## make model
    
    
#preprocess inputs(clip the label y)
    train_x=train_x[ind_train,:,:,:]
    train_x.shape=[train_x.shape[1],28,28,1]
    # input_shape = [train_x.shape[0],28,28]
    # inputs = Input(shape=input_shape)
    test_x=test_x[ind_test,:,:,:]
    test_x.shape=[test_x.shape[1],28,28,1]
    train_y=train_y[ind_train,:]
    train_y=train_y.reshape(train_y.shape[1],train_y.shape[2])
    test_y=test_y[ind_test,:]
    test_y=test_y.reshape(test_y.shape[1],test_y.shape[2])
    
    selec_convertor=np.zeros(([train_y.shape[1],num_class]))
    for i in range(num_class):
        tem=np.nonzero(selector)[0]
        selec_convertor[tem[i],i]=1 
    train_y=train_y@selec_convertor
    test_y=test_y@selec_convertor 
    model=tinyCNN()
    
    # n_hidden=32
    # n_out = num_class
    # # hidden1 = Dense(n_out, activation='sigmoid',kernel_initializer='RandomUniform')(inputs)
    # # hidden2=Dense(n_out, activation='softmax',kernel_initializer='RandomUniform')(hidden1)
    # # model = Model(inputs=inputs, outputs=hidden1)
    # model = keras.models.Sequential([
    #     keras.layers.Flatten(input_shape=(28, 28)),
    #     keras.layers.Dense(n_hidden, activation='relu'),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(n_out, activation='sigmoid')
    # ])
    # model.summary()
    # model= tf.keras.models.load_model("./weights/subs/sub"+str(ind_exp)+".h5")
    # model.load_weights("./weights/subs/sub"+str(i)+".h5")
    # print("Model loaded.")
#    log_dir=".\logs\fit\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    tbCallBack = TensorBoard(log_dir='.\logs', 
                 histogram_freq=0,
#                  batch_size=32,     
                 write_graph=False,  
                 write_grads=False, 
                 write_images=False,
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
    ##train model
    print(model.summary())
    model.load_weights("./weights/fashion_minist3.h5",by_name=True)
    # lr = 0.001
    epochs = 10
    checkpoint = keras.callbacks.ModelCheckpoint("./weights/subs/sub"+str(ind_exp)+".h5",
                verbose=0,
                save_best_only=True,
                monitor="val_accuracy",
                mode='max',
                save_weights_only=True
            )
    opt = tf.keras.optimizers.Adam(lr=1e-4, decay=0)
    
    # for i in range(1, 15):
    #     model.layers[i].trainable = False
        
    # 编译模型
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
     
    # 拟合数据
    model.fit(x=train_x, y=train_y,
    epochs=epochs,
    validation_data=(test_x, test_y),
    batch_size=16,
    callbacks=[ checkpoint],
    shuffle=True
    )

    # 模型评测
    test_loss, test_acc = model.evaluate(test_x, test_y,verbose=0)
 
    print('the model\'s test_loss is {} and test_acc is {}'.format(test_loss, test_acc))
    # opt = keras.optimizers.Adam(lr=1e-5, decay=0)
    # lr_reducer= ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
    #                                   cooldown=0, patience=10, min_lr=0.5e-6)
    # early_stopper= EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
    # print("save to "+str(ind_exp))
    # model_checkpoint= ModelCheckpoint("./weights/subs/sub"+str(ind_exp)+".h5", monitor="val_accuracy", save_best_only=True,
                                    # save_weights_only=False)
    # callbacks=[ model_checkpoint, tbCallBack]

    # model.compile(loss='categorical_crossentropy',
    #               optimizer=opt,
    #               metrics=['accuracy'])
    
    # train_x = train_x.astype('float32').reshape(train_x.shape[1],train_x.shape[2])
    # test_x = test_x.astype('float32').reshape(test_x.shape[1],test_x.shape[2])
    # hist = model.fit(train_x[:,:], train_y,
    #       batch_size=8,
    #       callbacks=[ model_checkpoint, tbCallBack],
    #       epochs=50,
    #       verbose=1,
    #       validation_data=(test_x[:,:], test_y),
    #       shuffle=True)
    # ## Score trained model.
    # scores = model.evaluate(test_x, test_y, verbose=0)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])
    
def main_eval():
    #eval the main model to get num_catch
    num_exp=5
    model= tf.keras.models.load_model(r'./weights/main2.h5')
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = \
        fashion_mnist.load_data()
    test_y=tf.keras.utils.to_categorical(test_labels)
    train_images, test_images = train_images / 255.0, test_images / 255.0
    data=test_images
    # data = scio.loadmat(r'C:\Users\wzs\Desktop\DFCN\data_feature_train.mat')['train']
    # data.shape=[data.shape[0],28,28]
    selector = scio.loadmat(r'./data/selector.mat')['selector']
# feature extraction
    layer_1 = tf.keras.backend.function([model.input], [model.output])
    num_catch=np.zeros(([data.shape[0],10]))
    count=0
    count2=0
    sele_dic={}
    '''dic---------------
        pred_labs------
            true_labs--'''
    for i in range(int(data.shape[0]/1)):
        f1 = layer_1([data[i,:].reshape(1,28*28)])[0]
        if np.argmax(test_y[i,:])!=np.argmax(f1):
            for j in range(num_exp):
                sel_mems=np.nonzero(selector[j,:])[0]
                if np.argmax(f1) in sel_mems:
                    if not np.argmax(f1) in sele_dic.keys():
                        sele_dic[np.argmax(f1)]={}
                        
                    if not np.argmax(test_y[i,:]) in sele_dic[np.argmax(f1)].keys():
                        sele_dic[np.argmax(f1)][np.argmax(test_y[i,:])]=[]
                    # print(np.argmax(f1))
                    # print(np.argmax(test_y[i,:]))
                    f2=np.sort(f1)[:,-2]
                    sele_dic[np.argmax(f1)][np.argmax(test_y[i,:])].append(f2/np.max(f1))
                    
    sele_dic_dis_mean={}
    sele_dic_dis_var={}
    for pred_labs in sele_dic:
        for true_labs in sele_dic[pred_labs]:
            if not pred_labs in sele_dic_dis_mean.keys():
                sele_dic_dis_mean[pred_labs]={}
                sele_dic_dis_var[pred_labs]={}
            sele_dic_dis_mean[pred_labs][true_labs]=np.mean(sele_dic[pred_labs][true_labs])
            sele_dic_dis_var[pred_labs][true_labs]=np.var(sele_dic[pred_labs][true_labs])
    sele_dis=[sele_dic_dis_mean,sele_dic_dis_var]
    with open("./data/sele_dis.pkl", 'wb') as f:
        pickle.dump(sele_dis, f)
    #         count=count+1
    #         print(str(i)+":\n")
    #         # print(f1)
    #         f1[:,np.argmax(f1)]=0
    #         print(np.argmax(f1))
    #         print(" ")
    #         print(np.argmax(test_y[i,:]))
    #         print("\n")
    #         if np.argmax(f1)==np.argmax(test_y[i,:]):
    #             count2=count2+1
    # print("wrong ans:"+str(count)+"\n")
    # print("wrong matches:"+str(count2)+"\n")
        # num_catch[i,:]=f1#record the num
    # scio.savemat('./data/num_catch.mat', {'num_catch':num_catch})
    # Download the data. The data is already divided into train and test.



def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image





def plot_confusion_matrix(cm, class_names):
  """
  Returns a matplotlib figure containing the plotted confusion matrix.
  Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
  """
  figure = plt.figure(figsize=(8, 8))
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
  plt.title("Confusion matrix")
  plt.colorbar()
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  # Normalize the confusion matrix.
  cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
  # print(cm.type)
  # Use white text if squares are dark; otherwise black.
  threshold = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    color = "white" if cm[i, j] > threshold else "black"
    plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return figure


def train_main():
    #-----------------------------------------------
    # The labels are integers representing classes.
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = \
        fashion_mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # train_images=train_images[0:20]
    # train_labels=train_labels[0:20]
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # train_x = scio.loadmat(r'C:\Users\wzs\Desktop\DFCN\data_feature_train.mat')['train']
    # test_x = scio.loadmat(r'C:\Users\wzs\Desktop\DFCN\data_feature_test.mat')['test']
    # train_images=train_x
    # test_images=test_x
    # Names of the integer classes, i.e., 0 -> T-short/top, 1 -> Trouser, etc.
    
    # class_names = ['0', '1', '2', '3', '4', 
    #     '5', '6', '7', '8', '9']
    print("Shape: ", train_images[0].shape)
    print("Shape_y: ", train_labels.shape)
    print("Label: ", train_labels[0], "->", class_names[train_labels[0]])
    
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        # keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = model.predict(test_images)
        test_pred = np.argmax(test_pred_raw, axis=1)
        
        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
        scio.savemat('./data/cm.mat', {'cm':cm})
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=class_names)
        cm_image = plot_to_image(figure)
          
        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
          tf.summary.image("Confusion Matrix", cm_image, step=epoch)
    logdir = "./logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Define the basic TensorBoard callback.
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,profile_batch = 100000000)
    file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
    model_checkpoint= tf.keras.callbacks.ModelCheckpoint("./weights/main2.h5", monitor="val_accuracy", save_best_only=True,
                                        save_weights_only=False)
    
    # Define the per-epoch callback.
    cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
    
    
    # Train the classifier.
    model.fit(
        train_images,
        train_labels,
        epochs=2,
        # verbose=0, # Suppress chatty output
        callbacks=[tensorboard_callback, cm_callback,model_checkpoint],
        validation_data=(test_images, test_labels),
    )
    
def evaluate():
    selector = scio.loadmat(r'./data/selector.mat')['selector']
    num_exp=5
    num_class=2
    model= tf.keras.models.load_model(r'./weights/main2.h5')
    sub_models=[]
    for i in range(num_exp):
        sub_models.append(tf.keras.models.load_model('./weights/subs/sub'+str(i)+'.h5'))
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = \
        fashion_mnist.load_data()
    data=test_images
    test_y=tf.keras.utils.to_categorical(test_labels)
    layer_1 = tf.keras.backend.function([model.input], [model.output])
    num_catch=np.zeros(([data.shape[0],10]))
    with open("./data/sele_dis.pkl", "rb") as f:
        sele_dis = pickle.load(f)
    sele_dic_dis_mean=sele_dis[0]
    sele_dic_dis_var=sele_dis[1]
    

    # main-------------------------------------------
    count=0
    for i in range(data.shape[0]):
        f1 = layer_1([data[i,:].reshape(1,28*28)])[0]
        pred_lab=np.argmax(f1)
        pred_prob_f2=np.sort(f1)[:,-2]/np.max(f1)
        for j in range(num_exp):
            sel_mems=np.nonzero(selector[j,:])[0]
            if np.argmax(f1) in sel_mems:
                ind=int(not(np.where(sel_mems==np.argmax(f1))[0][0]))
                true_lab=sel_mems[ind]
                if pred_prob_f2>sele_dic_dis_mean[pred_lab][true_lab]+sele_dic_dis_var[pred_lab][true_lab]:
                    sub_model=sub_models[j]
                    layer_sub = tf.keras.backend.function([sub_model.input], [sub_model.output])
                    f1_sub = layer_sub([data[i,:].reshape(1,28*28)])[0]
                    
                    selec_convertor=np.zeros(([10,num_class]))#transfer 10 to 2 classes
                    for i in range(num_class):
                        tem=np.nonzero(selector[j,:])[0]
                        selec_convertor[tem[i],i]=1 
                    f1_sub=f1_sub@selec_convertor.T
                    
                    pred_lab=np.argmax(f1_sub)
                    
        if pred_lab==np.argmax(test_y[i,:]):
                count=count+1
                
    print("wrong ans:"+str(count/10000)+"\n")
def main():
    # train_main()
    train_subs()
    # main_eval()
    # evaluate()
if __name__ == '__main__':
    main()