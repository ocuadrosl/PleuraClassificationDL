import cv2 as cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import matplotlib.pyplot as plt
from random import choices

import LoadDataset as ld

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227, 227))
    return image, label


# def LoadSet(inputDir, targetSet, size):
#     '''
#     Load "size" number of elementes randomly
#     '''
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
    nonPleura = ld.LoadSet(inputDir, "non_pleura", samplesSize)

    trainSet, validationSet, testSet, \
    trainLabels, validationLabels, testLabels = ld.SplitTrainValidationTest(pleura, nonPleura, 0.70, 0.15, 0.15)

    train_ds = tf.data.Dataset.from_tensor_slices((trainSet, trainLabels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validationSet, validationLabels))
    test_ds = tf.data.Dataset.from_tensor_slices((testSet, testLabels))

    # plt.figure(figsize=(20, 20))
    for i, (image, label) in enumerate(test_ds.take(5)):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(image)
        print(label.numpy())
        plt.title(CLASS_NAMES[label.numpy()])
        plt.axis('off')
    # plt.show()

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

    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                            input_shape=(227, 227, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.add(tf.keras.layers.Masking(mask_value=0., input_shape=(227, 227, 3)))

    root_logdir = os.path.join(os.curdir, "logs")
    
    
    def get_run_logdir():
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)
    
    
    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(learning_rate=0.001),
                  metrics=['accuracy'])
    model.summary()
    
    model.fit(train_ds, epochs=50, validation_data=validation_ds, validation_freq=1, callbacks=[tensorboard_cb])
    
    model.evaluate(test_ds)
