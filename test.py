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

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import scipy.io as scio
mnist = tf.keras.datasets.mnist
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."
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
    train_images=train_images[0:20]
    train_labels=train_labels[0:20]
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
def main():
    # train_main()
    main_eval()
if __name__ == '__main__':
    main()