import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from keras.callbacks import LearningRateScheduler

class Conv_Net:
    def __init__(self, kernel, stride, nets, names):
        self.kernel = kernel
        self.stride = stride
        self.nets = nets
        self.names = names
        self.models = {}
        self.histories={}

# Building a Convolution Neural Network with varying strides and kernels:
def get_models(self):

    # Stride Loop
    for i in range(2):
        model = [0] *self.nets
        # Kernel Loop
        for j in range(3):
            model[j] = Sequential()
            # 1st Convolution Layer
            model[j].add(Conv2D(24,kernel_size=self.kernel[j],strides=self.stride[i], padding='same',activation='relu',
                    input_shape=(28,28,1)))
            model[j].add(MaxPool2D())
            # 2nd Convolution Layer
            model[j].add(Conv2D(48,kernel_size=self.kernel[j], strides=self.stride[i], padding='same',activation='relu'))
            model[j].add(MaxPool2D())

            model[j].add(Flatten())
            model[j].add(Dense(256, activation='relu'))
            model[j].add(Dense(10, activation='softmax'))
            model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.models[i] = model

# train all the models and store hsitories for plotting.
def train(self, epochs):
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
    
    # training loop
    for i in range(2):
        history = [0] * self.nets
        model = self.models[i]
        for j in range(3):
            history[j] = model[j].fit(xTrain, yTrain, batch_size=1, epochs = epochs, 
                validation_data = (xValidate,yValidate),callbacks=[annealer], verbose=0)
            print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
            self.names[i][j],epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
        
        self.histories[i]=history

# plot the learning rate for all the strides and kernel size.
def plot_lr(self):
    styles = ['solid', 'dashed', 'dashdot']
    plt.figure(figsize=(15,5))
    for i in range(2):
        history = self.histories[i]
        for j in range(self.nets):
            plt.plot(history[j].history['val_accuracy'],linestyle=styles[j])
        plt.title('Training Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(self.names[i], loc='upper left')
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.show()


if '__name__' == '__main__':
    # load and split dataset into trainig and testing sets.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # One-hot encode the target variable.
    # y_train = utils.to_categorical(y_train, num_classes = 10)
    # y_test = utils.to_categorical(y_test, num_classes = 10)       

    # Split into training and validation set.
    xTrain, xValidate, yTrain, yValidate = train_test_split(x_train, y_train, test_size=5000, random_state=42)


    # create the CNN models:
    conv_net = Conv_Net([3,5,7], [1,2], 3, [['S1_K3', 'S1_K5', 'S1_K7'], ['S2_K3', 'S2_K5', 'S2_K7']])
    conv_net.get_models()

    # train models
    conv_net.train(20)

    # train the networks for 20 epochs: 
    conv_net.train(20)

    # plot and evaluate the accuracy for different strides and kernel size.
    conv_net.plot()


