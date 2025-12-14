from loadMnist import loadMnist 
import numpy as np
import os

# Ý tưởng: Sau khi thực hiện phép rút đặc trưng bằng phương pháp Histogram, 
#           với mỗi hình, ta thu được một array đặc trưng có kích thước 256. 
#          Tại đây ta sẽ thực hiện thu gọn array đặc trưng này xuống còn 64 phần tử 
#           bằng cách gom nhóm 4 phần tử liền kề nhau lại thành 1 phần tử bằng cách tính trung bình cộng. 

def anotherExtract(floatImgs): 
    n,m,p = floatImgs.shape 
    IntImgs = floatImgs.astype(int) # Vì mảng ban đầu là kiểu float cần chuyển sang kiểu int
    histogramArray = np.zeros((n, 256), dtype = int) # Chuẩn bị mảng 
    for i in range(n): 
        histogramArray[i] = np.bincount(IntImgs[i].ravel(), minlength = 256) # Rút đặc trưng bằng phương pháp Histogram
    histogramArray = histogramArray/ (m * p)  # Chuẩn hóa về [0,1] 
    myHistogramArray = histogramArray.reshape(n, 64, 4).mean(axis = 2).astype(float) # Gom 4 phần tử liền kề nhau thành 1 phần tử bằng trung bình cộng
    return myHistogramArray

# Load 4 mảng numpy 
dataFolder = '../data' 
trainImgs, trainLabels = loadMnist(dataFolder, kind='train')
testImgs, testLabels = loadMnist(dataFolder, kind='t10k')
processedTrainImgs = anotherExtract(trainImgs) 
processedTestImgs = anotherExtract(testImgs) 

# print("Kích thước đặc trưng Train:", processedTrainImgs.shape)
# print("Kích thước đặc trưng Test :", processedTestImgs.shape)

OUTPUT_DIR = os.path.join('..', 'resultFeatureExtract') 
train_save_path = os.path.join(OUTPUT_DIR, "trainAnother.npy")
test_save_path = os.path.join(OUTPUT_DIR, "testAnother.npy")

np.save(train_save_path, processedTrainImgs)
np.save(test_save_path, processedTestImgs)

