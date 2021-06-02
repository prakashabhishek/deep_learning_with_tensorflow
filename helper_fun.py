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


# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.
  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.
  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")

def compare_historys(original_history, new_history, initial_epochs=5):
    """
    Compares two model history objects.
    """
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()




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