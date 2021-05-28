import zipfile
import os
import numpy as np
import pandas as pd
import random
import pathlib
import datetime
import tensorflow as tf
print(f'Using tensorflow version: {tf.__version__}')

import tensorflow_hub as hub
from tensorflow.keras import Sequential, layers, losses, optimizers
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model and return 
    Precision, Recall, F1 score and Accuracy
    """
    accuracy = np.round(accuracy_score(y_true, y_pred) * 100, 2)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average= 'weighted')
    
    model_results = {"accuracy": accuracy,
                    "precision": np.round(precision*100, 2),
                    "recall": np.round(recall*100, 2),
                    "f1 score": np.round(f1_score*100,2)}
    
    return model_results



def unzip_data(file_path):
    """
    Unzips files to current directory
    """
    
    zip_ref = zipfile.ZipFile(file_path)
    zip_ref.extractall()
    zip_ref.close()


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates tensorboard callback to be monitor the model's performance
    """
    log_dir = os.path.join(dir_name, experiment_name, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir= log_dir)
    print(f'Saving tensorboard log files to {log_dir}')
    
    return tensorboard_callback



def plot_history(history):
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    plt.figure(figsize = (15,7))
    plt.subplot(1,2,1)
    plt.plot(train_loss, label = 'train loss')
    plt.plot(val_loss, label = 'val loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    
    plt.subplot(1,2,2)
    plt.plot(train_acc, label = 'train acc')
    plt.plot(val_acc, label = 'val acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()
    
    
def view_random_image(target_dir, target_class = None):
    """
    Displays random pictures from the target directory
    If target class is not given, then it will display
    rando images from any of the classes present in the 
    target directory else a random image from target class 
    will be displayed
    """
    
    data_dir = pathlib.Path(target_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    
   
    if not target_class:
        target_class = random.sample(list(class_names), 1)[0]

    
    target_folder = os.path.join(target_dir, target_class)
    

    image_path = random.sample(os.listdir(target_folder), 1)[0]
    img = mpimg.imread(os.path.join(target_folder, image_path))
    plt.imshow(img)
    plt.title(f'{target_class}: {image_path}')
    plt.axis('off')
    plt.show()
    
    print(f'Image shape: {img.shape}') # Show the shape of image
    
    
def prep_image_and_make_prediction(model, test_image_path, train_data_dir, img_shape = 244):
    """
    **This works for single image at time**
    
    Read an image from filename and turns into a tensor
    then reshapes it to (img_shape, img_shape, color_channel)
    """
    
    data_dir = pathlib.Path(target_dir)
    class_names = np.array(sorted([item.name for item in train_data_dir.glob('*')]))
    
    # Read in the image
    img = tf.io.read_file(test_image_path)

    # Decode the read file into tensor
    img = tf.image.decode_image(img)

    # Resize the image
    img = tf.image.resize(img, size = [img_shape, img_shape])
    
    # rescale the image 
    img = img/255
    
    if img.ndim != 4:
        img = tf.expand_dims(img, axis = 0)  # add 1 extra dimension for number of batches
        
    pred = model.predict(img)
    pred_class = class_names[np.argmax(pred)]
    
    plt.imshow(tf.squeeze(img))
    plt.title(f'Prediction: {pred_class} ({np.round(pred.max(), 2)})')
    plt.axis(False)
    plt.show()
    
    return 