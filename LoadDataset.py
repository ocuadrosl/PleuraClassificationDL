#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:06:36 2022

@author: oscar
"""

import cv2 as cv2
import os
import numpy as np
from random import choices

def LoadSet(inputDir, targetSet, size):
    '''
    Load "size" number of elementes randomly
    '''
    count = 0
    images = []
    for imageName in os.listdir(inputDir + '/' + targetSet):
        #print(imageName)
        image = cv2.imread(inputDir + '/' + targetSet + '/' + imageName, cv2.IMREAD_COLOR)
        images.append(image)
        count += 1
        #if count >= size:
        #    break

    #return sample(images, size)
    return choices(images, k=size)


def SplitTrainValidationTest(pleura, nonPleura, trainRate, validationRate, testRate):
    samplesSize = len(pleura)

    trainIndex = int(trainRate * samplesSize)
    validationIndex = trainIndex + int(validationRate * samplesSize)

    trainSet = pleura[:trainIndex] + nonPleura[:trainIndex]
    trainLabels = np.hstack((np.zeros(trainIndex, dtype=np.int8), np.ones(trainIndex, dtype=np.int8)))

    validationSet = pleura[trainIndex: validationIndex] + nonPleura[trainIndex:validationIndex]
    validationLabels = np.hstack(
        (np.zeros(validationIndex - trainIndex, dtype=np.int8), np.ones(validationIndex - trainIndex, dtype=np.int8)))
    print(len(validationLabels), len(validationSet))

    testSet = pleura[validationIndex:] + nonPleura[validationIndex:]
    testLabels = np.hstack(
        (np.zeros(samplesSize - validationIndex, dtype=np.int8), np.ones(samplesSize - validationIndex, dtype=np.int8)))
    print(len(testLabels), len(testSet))

    return trainSet, validationSet, testSet, trainLabels, validationLabels, testLabels



