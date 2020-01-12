#!/usr/bin/env python
# encoding: utf-8




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

import sklearn.metrics
mnist = tf.keras.datasets.mnist
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."
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
    f = h5py.File(r'C:\Users\wzs\Desktop\DFCN\train_dataset.h5','r')                             
    train_y_orig = f['train_labels'][:] 
    train_y=to_categorical(train_y_orig) 
    f = h5py.File(r'C:\Users\wzs\Desktop\DFCN\test_dataset.h5','r')
    test_y_orig = f['test_labels'][:] 
    test_y=to_categorical(test_y_orig)
    
#train sub models
    dic_sub ={}
    for i in range(selector.shape[0]):
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
#preprocess inputs(clip the label y)
    train_x=train_x[ind_train,:]
    test_x=test_x[ind_test,:]
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

    n_hidden=100
    n_out = num_class
    hidden1 = Dense(n_out, activation='softmax',kernel_initializer='RandomUniform')(inputs)
    # hidden2=Dense(n_out, activation='softmax',kernel_initializer='RandomUniform')(hidden1)
    model = Model(inputs=inputs, outputs=hidden1)
    model.summary()
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
    opt = keras.optimizers.SGD(lr=1e-4, decay=0)
    # model.load_weights("weights/subs/sub"+str(i)+".h5")
    # print("Model loaded.")

    lr_reducer= ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                      cooldown=0, patience=10, min_lr=0.5e-6)
    early_stopper= EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
    print("save to "+str(ind_exp))
    model_checkpoint= ModelCheckpoint("./weights/subs/sub"+str(ind_exp)+".h5", monitor="val_acc", save_best_only=False,
                                    save_weights_only=False)
    callbacks=[early_stopper, model_checkpoint, tbCallBack]

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    
    train_x = train_x.astype('float32').reshape(train_x.shape[1],train_x.shape[2])
    test_x = test_x.astype('float32').reshape(test_x.shape[1],test_x.shape[2])
    hist = model.fit(train_x[:,:], train_y,
          batch_size=32,
          callbacks=callbacks,
          epochs=5,
          verbose=0,
          validation_data=(test_x[:,:], test_y),
          shuffle=True)
    ## Score trained model.
    scores = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    
def main_eval():
    #eval the main model to get num_catch
    model= tf.keras.models.load_model(r'./weights/main2.h5')
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = \
        fashion_mnist.load_data()
    data=test_images
    # data = scio.loadmat(r'C:\Users\wzs\Desktop\DFCN\data_feature_train.mat')['train']
    # data.shape=[data.shape[0],28,28]
 
# feature extraction
    layer_1 = tf.keras.backend.function([model.input], [model.output])
    num_catch=np.zeros(([data.shape[0],10]))
    for i in range(data.shape[0]):
        
        f1 = layer_1([data[i,:].reshape(1,28*28)])[0]
        if i%100==0:
            print(i)
        num_catch[i,:]=f1#record the num
    scio.savemat('./data/num_catch.mat', {'num_catch':num_catch})
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
    # train_images=train_images[0:20]
    # train_labels=train_labels[0:20]
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
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
        keras.layers.Dense(10, activation='softmax')
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
    model= tf.keras.models.load_model(r'./weights/main2.h5')
    sub_models=[]
    for i in range(num_exp):
        sub_models.append(tf.keras.models.load_model('./weights/subs/sub'+str(i)+'.h5'))
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = \
        fashion_mnist.load_data()
    data=test_images
    layer_1 = tf.keras.backend.function([model.input], [model.output])
    num_catch=np.zeros(([data.shape[0],10]))
    for i in range(data.shape[0]):
        f1 = layer_1([data[i,:].reshape(1,28*28)])[0]
        # num_catch[i,:]=f1#record the num
        if :
            
def main():
    # train_main()
    train_subs()
    # evaluate()
if __name__ == '__main__':
    main()