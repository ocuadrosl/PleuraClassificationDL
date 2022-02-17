# import the necessary packages
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

import tensorflow as tf

import cv2 as cv2
import os
from random import choices
import numpy as np
from tensorflow import keras
import time

import LoadDataset as ld

class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # the shortcut branch of the ResNet module should be
        # initialize as the input (identity) data
        shortcut = data
           
        # the first block of the ResNet module are the 1x1 CONVs
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)
         
        # the second block of the ResNet module are the 3x3 CONVs
        bn2= BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act2)
         
         
        # the third block of the ResNet module is another set of 1x1 CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)
         
        # if we are to reduce the spatial size, apply a CONV layer to the shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)
        
        # add together the shortcut and the final CONV
        x = add([conv3, shortcut])
        
        # return the addition as the output of the ResNet module
        return x

    @staticmethod   
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        # initialize the input shape to be "channels last" and the
        # channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1            
            
        # set the input and apply BN
        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,momentum=bnMom)(inputs)

        # apply CONV => BN => ACT => POOL to reduce spatial size
        x = Conv2D(filters[0], (5, 5), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)


        # loop over the number of stages
        for i in range(0, len(stages)):
            # initialize the stride, then apply a residual module
            # used to reduce the spatial size of the input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

            # loop over the number of layers in the stage
            for j in range(0, stages[i] - 1):
                # apply a ResNet module
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)
                
                
        # apply BN => ACT => POOL
        x = BatchNormalization(axis=chanDim, epsilon=bnEps,	momentum=bnMom)(x)
        
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)
        
        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)
	
        # create the model
        model = Model(inputs, x, name="resnet")
        
	
        # return the constructed network architecture
        return model
     
        



def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227, 227))
    return image, label


# def LoadSet(inputDir, targetSet, size):
#     count = 0
#     images = []
#     for imageName in os.listdir(inputDir + '/' + targetSet):
#         #print(imageName)
#         image = cv2.imread(inputDir + '/' + targetSet + '/' + imageName, cv2.IMREAD_COLOR)
#         images.append(image)
#         count += 1
#         #if count >= size:
#         #    break

#     #return sample(images, size)
#     return choices(images, k=size)



# def SplitTrainValidationTest(pleura, nonPleura, trainRate, validationRate, testRate):
#     samplesSize = len(pleura)

#     trainIndex = int(trainRate * samplesSize)
#     validationIndex = trainIndex + int(validationRate * samplesSize)

#     trainSet = pleura[:trainIndex] + nonPleura[:trainIndex]
#     trainLabels = np.hstack((np.zeros(trainIndex, dtype=np.int8), np.ones(trainIndex, dtype=np.int8)))

#     validationSet = pleura[trainIndex: validationIndex] + nonPleura[trainIndex:validationIndex]
#     validationLabels = np.hstack(
#         (np.zeros(validationIndex - trainIndex, dtype=np.int8), np.ones(validationIndex - trainIndex, dtype=np.int8)))
#     print(len(validationLabels), len(validationSet))

#     testSet = pleura[validationIndex:] + nonPleura[validationIndex:]
#     testLabels = np.hstack(
#         (np.zeros(samplesSize - validationIndex, dtype=np.int8), np.ones(samplesSize - validationIndex, dtype=np.int8)))
#     print(len(testLabels), len(testSet))

#     return trainSet, validationSet, testSet, trainLabels, validationLabels, testLabels

if __name__ == "__main__":
    
        
    inputDir = "/home/oscar/data/biopsy/dataset_3/slices_227_RGB"
    
    CLASS_NAMES = ['pleura', 'non_pleura']
    
    # load samples pleura and non pleura
    
    samplesSize = 4000 #4000  # for pleura and non_pleura
    
    pleura = ld.LoadSet(inputDir, "pleura", samplesSize)
    nonPleura =ld. LoadSet(inputDir, "non_pleura", samplesSize)
    
    trainSet, validationSet, testSet, \
    trainLabels, validationLabels, testLabels = ld.SplitTrainValidationTest(pleura, nonPleura, 0.70, 0.15, 0.15)
    
    train_ds = tf.data.Dataset.from_tensor_slices((trainSet, trainLabels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validationSet, validationLabels))
    test_ds = tf.data.Dataset.from_tensor_slices((testSet, testLabels))
    
    
    
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
    
    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", validation_ds_size)
    
    train_ds = (train_ds
                .map(process_images)
                .shuffle(buffer_size=train_ds_size)
                .batch(batch_size=32, drop_remainder=True))
    test_ds = (test_ds
               .map(process_images)
               .shuffle(buffer_size=test_ds_size)
               .batch(batch_size=32, drop_remainder=True))
    validation_ds = (validation_ds
                     .map(process_images)
                     .shuffle(buffer_size=validation_ds_size)
                     .batch(batch_size=32, drop_remainder=True))




root_logdir = os.path.join(os.curdir, "logs")
    
    
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
        
        
#build model
resNet = ResNet()
model = resNet.build(227, 227, 3, 2, (3, 4, 6), (64, 128, 256, 512))
    
#model.add(tf.keras.layers.Masking(mask_value=0., input_shape=(227, 227, 3)))
    
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
model.summary()    
    
model.fit(train_ds, epochs=50, validation_data=validation_ds, validation_freq=1, callbacks=[tensorboard_cb])
    
model.evaluate(test_ds)
        
            
            
            
            