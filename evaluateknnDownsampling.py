import numpy as np
import os
import time
from knnClassifier import kNNPredictLabel
from knnClassifier import loadFeature


# Load train và test bằng Downsampling
featureExtractDir = 'resultFeatureExtract' 
trainDownsampling = loadFeature('downsampling', 'train')
testDownsampling = loadFeature('downsampling', 'test')


# Giá trị k hiện tại
kValue = 5


# Cài đặt cấu hình cho việc trả kết quả
featureMethod = 'downsampling'
outputDir = os.path.join('resultkNN', f'result{featureMethod}') 

print(f"Kích thước đặc trưng Train: {trainDownsampling.shape}")
print(f"Kích thước đặc trưng Test : {testDownsampling.shape}")

N_test = testDownsampling.shape[0]

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
print(f"\n-> Đã lưu mảng nhãn kết quả vào file: {output_filepath}")
