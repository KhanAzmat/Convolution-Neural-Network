import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utils import unpickle, reshape_transpose, normalise
from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, MaxPool2D


class CNN_CIFAR:
    def __init__(self, kernel, shape, metrics, names):
        self.kernel = kernel
        self.shape = shape
        self.metrics = metrics
        self.history = {}
        self.model = [0]*len(kernel)
        self.names = names
        


    # define models for varying kernel size.
    '''
    Model Summary for kernel size 3
        Model: "sequential"
        _________________________________________________________________
        Layer (type)                Output Shape              Param #   
        =================================================================
        conv2d (Conv2D)             (None, 32, 32, 32)        896       
                                                                        
        batch_normalization (Batch  (None, 32, 32, 32)        128       
        Normalization)                                                  
                                                                        
        conv2d_1 (Conv2D)           (None, 32, 32, 32)        9248      
                                                                        
        batch_normalization_1 (Bat  (None, 32, 32, 32)        128       
        chNormalization)                                                
                                                                        
        max_pooling2d (MaxPooling2  (None, 16, 16, 32)        0         
        D)                                                              
                                                                        
        dropout (Dropout)           (None, 16, 16, 32)        0         
                                                                        
        conv2d_2 (Conv2D)           (None, 16, 16, 64)        18496     
                                                                        
        batch_normalization_2 (Bat  (None, 16, 16, 64)        256       
        chNormalization)                                                
                                                                        
        conv2d_3 (Conv2D)           (None, 16, 16, 64)        36928     
                                                                        
        batch_normalization_3 (Bat  (None, 16, 16, 64)        256       
        chNormalization)                                                
                                                                        
        max_pooling2d_1 (MaxPoolin  (None, 8, 8, 64)          0         
        g2D)                                                            
                                                                        
        dropout_1 (Dropout)         (None, 8, 8, 64)          0         
                                                                        
        conv2d_4 (Conv2D)           (None, 8, 8, 128)         73856     
                                                                        
        batch_normalization_4 (Bat  (None, 8, 8, 128)         512       
        chNormalization)                                                
                                                                        
        conv2d_5 (Conv2D)           (None, 8, 8, 128)         147584    
                                                                        
        batch_normalization_5 (Bat  (None, 8, 8, 128)         512       
        chNormalization)                                                
                                                                        
        max_pooling2d_2 (MaxPoolin  (None, 4, 4, 128)         0         
        g2D)                                                            
                                                                        
        dropout_2 (Dropout)         (None, 4, 4, 128)         0         
                                                                        
        flatten (Flatten)           (None, 2048)              0         
                                                                        
        dense (Dense)               (None, 128)               262272    
                                                                        
        dropout_3 (Dropout)         (None, 128)               0         
                                                                        
        dense_1 (Dense)             (None, 10)                1290      
                                                                        
        =================================================================
        Total params: 552362 (2.11 MB)
        Trainable params: 551466 (2.10 MB)
        Non-trainable params: 896 (3.50 KB)
    '''
    def create_models(self):
        for j in range(2):
            self.model[j] = Sequential()
            # Convolutional Layer
            self.model[j].add(Conv2D(filters=32, kernel_size=kernel[j] ,  input_shape=self.shape, activation='relu', padding='same'))
            self.model[j].add(BatchNormalization())
            self.model[j].add(Conv2D(filters=32, kernel_size=kernel[j] ,  input_shape=self.shape, activation='relu', padding='same'))
            self.model[j].add(BatchNormalization())
            # Pooling layer
            self.model[j].add(MaxPool2D())
            # Dropout layers
            self.model[j].add(Dropout(0.25))

            self.model[j].add(Conv2D(filters=64, kernel_size=kernel[j] ,  input_shape=self.shape, activation='relu', padding='same'))
            self.model[j].add(BatchNormalization())
            self.model[j].add(Conv2D(filters=64, kernel_size=kernel[j] ,  input_shape=self.shape, activation='relu', padding='same'))
            self.model[j].add(BatchNormalization())
            self.model[j].add(MaxPool2D())
            self.model[j].add(Dropout(0.25))

            self.model[j].add(Conv2D(filters=128, kernel_size=kernel[j] ,  input_shape=self.shape, activation='relu', padding='same'))
            self.model[j].add(BatchNormalization())
            self.model[j].add(Conv2D(filters=128, kernel_size=kernel[j] ,  input_shape=self.shape, activation='relu', padding='same'))
            self.model[j].add(BatchNormalization())
            self.model[j].add(MaxPool2D())
            self.model[j].add(Dropout(0.25))

            self.model[j].add(Flatten())
            self.model[j].add(Dense(128, activation='relu'))
            self.model[j].add(Dropout(0.25))
            self.model[j].add(Dense(10, activation='softmax'))
            self.model[j].compile(loss='categorical_crossentropy', optimizer='adam', metrics=self.metrics)

        
    # to train the models.
    def train(self, epochs, xTrain, yTrain, xValidate, yValidate):
        for j in range(2):
            self.history[j] = self.model[j].fit(xTrain, yTrain, batch_size=5, epochs = epochs, 
                validation_data = (xValidate,yValidate), verbose=0)
            print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
            self.names[j],epochs,max(self.history[j].history['accuracy']),max(self.history[j].history['val_accuracy']) ))

    # plot and evalute the training accuracy for each kernel.
    def plot(self):
        # Plot learning rate:
        styles = ['solid', 'dashed']
        plt.figure(figsize=(15,5))

        for j in range(2):
            plt.plot(self.history[j].history['val_accuracy'],linestyle=styles[j])
        plt.title('Training Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(self.names, loc='upper left')
        axes = plt.gca()
        axes.set_ylim([0,1])
        plt.show()

    # test the trained model
    def test(self,x_test, model_num):
        results = np.zeros( (x_test.shape[0],10) ) 
        results = results + self.model[model_num].predict(x_test, verbose=0)
        return np.argmax(results,axis = 1)

    # to visualise the incorrectly identified images.
    def plot_images(self,x_test, pred, imageLabels):
        plt.figure(figsize=(19,6))
        plt.tight_layout() 
        count = 1
        for i in range(31, 61, 1):
            plt.subplot(3,10,count)
            plt.imshow(x_test[incorrect[i]])
            plt.title(f"Predict: {imageLabels[pred[incorrect[i]]]}")
            plt.axis('off')
            count +=1
        plt.show()

    # to show intermediate feature maps.
    def feature_maps(self, data, layers, model_num):
        intermediate_layer_model = tf.keras.Model(inputs=self.model[0].input, outputs=self.model[model_num].get_layer(layers[1]).output)

        # Get the intermediate feature maps
        intermediate_feature_maps = intermediate_layer_model.predict(data, verbose=0)

        # Plot the feature maps
        plt.figure(figsize=(10, 10))

        for j in range(intermediate_feature_maps.shape[3]):
            plt.subplot(8, 4, j + 1)  # Adjust the subplot layout based on the number of feature maps
            plt.imshow(intermediate_feature_maps[0, :, :, j])
            plt.axis('off')
        plt.show()




if '__name__' == '__main__':

    # for printing all columns and rows
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file = './cifar-10-batches-py/batches.meta'
    
    # get dataset
    data = unpickle(file)

    # Getting label strings for the images :
    imageLabels = data[b'label_names']
    imageLabels = [item.decode('utf-8') for item in imageLabels]

    # loading training data
    data_train = []
    for i in range(1,6):
        file = './cifar-10-batches-py/data_batch_{}'.format(i)
        data = unpickle(file)
        data_train.append(data)

    # split feature and target
    x_train = []
    y_train = []

    for i in range(5):
        label = np.array(data_train[i][b'labels']).reshape(10000,1)
        x_train.append(data_train[i][b'data'])
        y_train.append(label)

    #Consolidated training set
    x_train = np.concatenate((x_train[0], x_train[1], x_train[2], x_train[3], x_train[4]), axis=0) 
    y_train = np.concatenate((y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]), axis=0)

    # Test Set
    file = './cifar-10-batches-py/test_batch'
    data = unpickle(file)

    y_test = np.array(data[b'labels']).reshape(10000,1)
    x_test = data[b'data']

    # reshape and transpose the images
    x_train = reshape_transpose(x_train)
    x_test = reshape_transpose(x_test)

    # one-hot encode the target
    # y_train = utils.to_categorical(y_train, num_classes = 10)
    # y_test = utils.to_categorical(y_test, num_classes = 10)

    # split into training and validation set
    xTrain, xValidate, yTrain, yValidate = train_test_split(x_train, y_train, test_size=5000, random_state=42)

    # normalise the input features
    xTrain = normalise(xTrain)
    xValidate = normalise(xValidate)
    x_test = normalise(x_test)

    # define the input shape and the kernel size
    shape = (32, 32, 3)
    kernel = [3,5]
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    names = ['S1_K3', 'S1_K5']

    conv_net = CNN_CIFAR(kernel, shape, metrics, names)

    # create models to evaluate
    conv_net.create_models()

    # train the models on the training dataset
    conv_net.train(50, xTrain, yTrain, xValidate, yValidate)

    # track the training accuracy
    conv_net.plot()

    # test the model
    pred = conv_net.test(x_test, 0)
    

    # get the incorrectly identified images.
    incorrect =[]
    correct = []
    for i in range(10000):
        if pred[i] != np.argmax(y_test[i]):
            incorrect.append(i)
        else:
            correct.append(i)

    # to show the incorrectly identified images
    conv_net.plot_images(x_test, pred, imageLabels)

    # to show intermediate feature maps.
    inter_layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5']
    img_array = x_test[1].reshape((1, 32, 32, 3))
    conv_net.feature_maps(img_array, inter_layers, 0)





