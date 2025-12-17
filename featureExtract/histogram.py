try:
    from featureExtract.loadMnist import loadMnist
except ImportError:
    from loadMnist import loadMnist
import numpy as np
import os
from skimage.feature import hog

def histogramExtract(floatImgs):
    n,m,p = floatImgs.shape
    featureList = []
    for i in range(n):
        img = floatImgs[i]
        hogVector = hog(img,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        orientations=9)
        featureList.append(hogVector)
    featureArray = np.array(featureList) 
    newLen = featureArray.shape[1] // 4
    featureArray = featureArray[:, :newLen * 4]
    featureArray = featureArray.reshape(n, newLen, 4).mean(axis=2)
    return featureArray.astype(float)

try:
    dataFolder = 'data'
except:
    dataFolder = '../data'

trainImgs, trainLabels = loadMnist(dataFolder, kind='train')
testImgs,  testLabels  = loadMnist(dataFolder, kind='t10k')

processedTrainImgs = histogramExtract(trainImgs)
processedTestImgs  = histogramExtract(testImgs)

# print("Kích thước đặc trưng Train:", processedTrainImgs.shape)
# print("Kích thước đặc trưng Test :", processedTestImgs.shape)

try:
    OUTPUT_DIR = os.path.join('resultFeatureExtract')
except:
    OUTPUT_DIR = os.path.join('..', 'resultFeatureExtract') 

train_save_path = os.path.join(OUTPUT_DIR, "trainHistogram.npy")
test_save_path = os.path.join(OUTPUT_DIR, "testHistogram.npy")

np.save(train_save_path, processedTrainImgs)
np.save(test_save_path, processedTestImgs)

