from loadMnist import loadMnist 
import numpy as np 

# Load 4 mảng numpy (bắt buộc)
dataFolder = 'data' 
trainImgs, trainLabels = loadMnist(dataFolder, kind='train')
testImgs, testLabels = loadMnist(dataFolder, kind='t10k')

# Kiểm tra xem các hàm có đọc đúng chưa (chỉ là bước kiểm thử)
print(f"1. trainImgs shape:   {trainImgs.shape}")   # Mong đợi: (60000, 28, 28)
print(f"2. trainLabels shape: {trainLabels.shape}") # Mong đợi: (60000,)
print(f"3. testImgs shape:    {testImgs.shape}")    # Mong đợi: (10000, 28, 28)
print(f"4. testLabels shape:  {testLabels.shape}")  # Mong đợi: (10000,)
