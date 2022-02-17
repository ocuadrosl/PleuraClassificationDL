#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 11:32:03 2021

@author: oscar
"""

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import sys


def SplitImage(image, tileSize):
    """
        Split image into tiles of size tileSize
    """

    height, width, _ = image.shape
    #print(image.shape)

    tiles = []
    positions = []
    maxMultHeight = height - (height % tileSize)
    maxMultWidth = width - (width % tileSize)
    # print(maxMultHeight, maxMultWidth)
    for i in range(0, maxMultHeight, tileSize):
        for j in range(0, maxMultWidth, tileSize):
            # yield image[i:i+tileSize, j:j+tileSize]
            positions.append(np.asarray((i, i + tileSize, j, j + tileSize)))
            tiles.append(image[i:i + tileSize, j:j + tileSize])
            # print(image[i:i+tileSize, j:j+tileSize])

    #lastTile = image[maxMultHeight:height, maxMultWidth:width]
    #if lastTile.shape[0] > 0 and lastTile.shape[1] > 0:
    #    tiles.append(lastTile)
    #    positions.append(np.asarray((maxMultHeight, height, maxMultWidth, width)))
    return tiles, positions


def MaskInputTile(imageTile, maskTile):
    imageTile[maskTile == 0] = 0
    return imageTile


def CreateDataSet():
    None

if __name__ == "__main__":

    inputDir = "/home/oscar/data/biopsy/dataset_3"
    outputDir = "/home/oscar/data/biopsy/dataset_3/slices_227_RGB"
    boundaryDataSet = "erode_radius_30"
    targetSet = 'test'
    dataset = pd.DataFrame()
    tile_size = 227  # tiles
    minTileDataRate = 0.1
    colorType = 'images'

    for imageName in os.listdir(inputDir + '/'+colorType+'/' + targetSet):

        print(imageName.split(".")[0])
        inputImage = cv2.imread(inputDir + '/'+colorType+'/' + targetSet + '/' + imageName, cv2.IMREAD_COLOR)
        inputPleuraMask = cv2.imread(inputDir + "/masks/pleura/" + imageName, cv2.IMREAD_COLOR)
        inputNonPleuraMask = cv2.imread(inputDir + "/masks/non_pleura/" + imageName, cv2.IMREAD_COLOR)

        imageTiles, positions = SplitImage(inputImage, tile_size)

        pleuraTiles, positions = SplitImage(inputPleuraMask, tile_size)

        nonPleuraTiles, positions = SplitImage(inputNonPleuraMask, tile_size)

        tileNr = 0
        for (imageTile, pleuraTile, nonPleuraTile) in zip(imageTiles, pleuraTiles, nonPleuraTiles):

            #tileName = outputDir + "/" + imageName.split(".")[0] + "_" + str(tileNr) + ".jpg"

            if np.sum(pleuraTile != 0) >= (pleuraTile.shape[0] * pleuraTile.shape[1] * minTileDataRate): # no black tiles

                tileName = outputDir + "/pleura/" + imageName.split(".")[0] + "_" + str(tileNr) + ".jpg"
                #print("Pleura")
                cv2.imwrite(tileName, MaskInputTile(imageTile, pleuraTile))
                tileNr += 1

            elif np.sum(nonPleuraTile != 0) >= (nonPleuraTile.shape[0] * nonPleuraTile.shape[1] * minTileDataRate):

                tileName = outputDir + "/non_pleura/" + imageName.split(".")[0] + "_" + str(tileNr) + ".jpg"
                #print("Non pleura")
                cv2.imwrite(tileName, MaskInputTile(imageTile,nonPleuraTile))
                tileNr += 1

