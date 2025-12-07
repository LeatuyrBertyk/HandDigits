from loadMnist import loadMnist
import numpy as np
from skimage.feature import hog
def RutDacTrung_HOG(FloatImgs):
    n= FloatImgs.shape
    FeatureList = []
    for i in range(n):
        img = FloatImgs[i]
        hogVector = hog(img,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        orientations=9)
        FeatureList.append(hogVector)
    FeatureArray = np.array(FeatureList) 
    newLen = FeatureArray.shape[1] // 4
    FeatureArray = FeatureArray[:, :newLen * 4]
    FeatureArray = FeatureArray.reshape(n, newLen, 4).mean(axis=2)
    return FeatureArray.astype(float)
datafolder = 'data'
TrainImgs, TrainLabels = loadMnist(datafolder, kind='train')
TestImgs,  TestLabels  = loadMnist(datafolder, kind='t10k')
ProcessedTrainImgs = RutDacTrung_HOG(TrainImgs)
ProcessedTestImgs  = RutDacTrung_HOG(TestImgs)
print("Kích thước đặc trưng Train:", ProcessedTrainImgs.shape)
print("Kích thước đặc trưng Test :", ProcessedTestImgs.shape)
