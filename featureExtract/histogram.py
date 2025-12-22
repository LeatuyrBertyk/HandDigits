try:
    from featureExtract.loadMnist import loadMnist
except ImportError:
    from loadMnist import loadMnist

import numpy as np
import os

def histogramExtract(imgs):
    """
    imgs: (N, 28, 28), pixel 0-255
    return: (N, 256) histogram đã chuẩn hóa
    """
    n, h, w = imgs.shape
    features = np.zeros((n, 256), dtype=float)

    for i in range(n):
        img = imgs[i]
        hist = np.zeros(256)

        for pixel in img.flatten():
            hist[pixel] += 1

        hist = hist / (h * w)
        features[i] = hist

    return features

try:
    dataFolder = 'data'
except:
    dataFolder = '../data'

trainImgs, trainLabels = loadMnist(dataFolder, kind='train')
testImgs,  testLabels  = loadMnist(dataFolder, kind='t10k')

print("Train images shape:", trainImgs.shape)
print("Test images shape :", testImgs.shape)

processedTrainImgs = histogramExtract(trainImgs)
processedTestImgs  = histogramExtract(testImgs)

print("Train feature shape:", processedTrainImgs.shape) 
print("Test feature shape :", processedTestImgs.shape) 

try:
    OUTPUT_DIR = os.path.join('resultFeatureExtract')
except:
    OUTPUT_DIR = os.path.join('..', 'resultFeatureExtract')

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_save_path = os.path.join(OUTPUT_DIR, "trainHistogram.npy")
test_save_path  = os.path.join(OUTPUT_DIR, "testHistogram.npy")

np.save(train_save_path, processedTrainImgs)
np.save(test_save_path, processedTestImgs)
