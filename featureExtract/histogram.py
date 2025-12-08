from featureExtract.loadMnist import loadMnist
import numpy as np
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

datafolder = 'data'
trainImgs, trainLabels = loadMnist(datafolder, kind='train')
testImgs,  testLabels  = loadMnist(datafolder, kind='t10k')

processedTrainImgs = histogramExtract(trainImgs)
processedTestImgs  = histogramExtract(testImgs)

# print("Kích thước đặc trưng Train:", processedTrainImgs.shape)
# print("Kích thước đặc trưng Test :", processedTestImgs.shape)
