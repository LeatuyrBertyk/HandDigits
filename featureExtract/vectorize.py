from featureExtract.loadMnist import loadMnist 
import numpy as np

# Load 4 mảng numpy 
dataFolder = 'data' 
trainImgs, trainLabels = loadMnist(dataFolder, kind='train')
testImgs, testLabels = loadMnist(dataFolder, kind='t10k')

import numpy as np
import os

def vectorizeExtract(images):
    if images.ndim != 3:
        raise ValueError("Input phải có dạng (N, 28, 28)")

    numSamples = images.shape[0]

    # Flatten
    vectors = images.reshape(numSamples, 784).astype(np.float32) / 255.0

    # Binarize
    binaryVectors = (vectors >= 0.5).astype(np.uint8)
    
    return binaryVectors


trainFeatures = vectorizeExtract(trainImgs)
testFeatures = vectorizeExtract(testImgs)

# # In kết quả
# print("Train features shape:", trainFeatures.shape)
# print("Test features shape:", testFeatures.shape)
# 
# # Lưu để dùng cho KNN
# np.save("train_vector.npy", trainFeatures)
# np.save("test_vector.npy", testFeatures)

