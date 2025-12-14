import numpy as np
import os
import time
from featureExtract.loadMnist import loadMnist
from featureExtract.downsampling import downsamplingExtract
from knnClassifier import kNNPredictLabel


dataFolder = 'data' 
kValue = 5
featureMethod = 'downsampling'
outputDir = os.path.join('resultkNN', f'result{featureMethod}') 

trainImgs, trainLabels = loadMnist(dataFolder, kind='train')
testImgs, testLabels = loadMnist(dataFolder, kind='t10k')
trainFeatures = np.array([downsamplingExtract(img) for img in trainImgs])
testFeatures = np.array([downsamplingExtract(img) for img in testImgs])
print(f"Kích thước đặc trưng Train: {trainFeatures.shape}")
print(f"Kích thước đặc trưng Test : {testFeatures.shape}")

N_test = testFeatures.shape[0]
assignedLabels = np.zeros(N_test, dtype=int)
startTime = time.time()
for i in range(N_test):
    assignedLabels[i] = kNNPredictLabel(featureMethod, i, k=kValue)
endTime = time.time()
elapsedTime = endTime - startTime
print(f"Thời gian chạy: {elapsedTime:.2f} giây")

output_filename = f"{featureMethod}_k{kValue}_labels.npy"
output_filepath = os.path.join(outputDir, output_filename)
    
np.save(output_filepath, assignedLabels)
print(f"\n-> Đã lưu mảng nhãn kết quả vào file: **{output_filepath}**")
