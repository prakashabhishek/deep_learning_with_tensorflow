import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

import tensorflow as tf
print(f'Using tensorflow version: {tf.__version__}')

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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