import numpy as np
from scipy.spatial.distance import cdist  # Hỗ trợ hàm tính khoảng cách

from featureExtract.loadMnist import loadMnist 
# from featureExtract.vectorize import vectorizeExtract
from featureExtract.histogram import histogramExtract 
from featureExtract.downsampling import downsamplingExtract 
from featureExtract.another import anotherExtract 

dataFolder = 'data'
trainImgs, trainLabels = loadMnist(dataFolder, kind='train')
testImgs, testLabels = loadMnist(dataFolder, kind='t10k')

# Load các mảng train và test của từng phép rút đặc trưng

# trainVectorize = vectorizeExtract(trainImgs)
# testVectorize = vectorizeExtract(testImgs)
# print(f"Kích thước đặc trưng Vectorize (Train): {trainVectorize.shape}")
# print(f"Kích thước đặc trưng Vectorize (Test): {testVectorize.shape}")

trainHistogram = histogramExtract(trainImgs)
testHistogram = histogramExtract(testImgs)
# print(f"Kích thước đặc trưng Histogram (Train): {trainHistogram.shape}")
# print(f"Kích thước đặc trưng Histogram (Test ): {testHistogram.shape}\n")

trainDownsampling = np.array([downsamplingExtract(img) for img in trainImgs])
testDownsampling = np.array([downsamplingExtract(img) for img in testImgs])
# print(f"Kích thước đặc trưng Downsampling (Train): {trainDownsampling.shape}")
# print(f"Kích thước đặc trưng Downsampling (Test ): {testDownsampling.shape}\n")

trainAnother = anotherExtract(trainImgs)
testAnother = anotherExtract(testImgs)
# print(f"Kích thước đặc trưng Hist. Intensity (Train): {trainAnother.shape}")
# print(f"Kích thước đặc trưng Hist. Intensity (Test ): {testAnother.shape}\n")
 


featureData = {
    # 'vectorize': {'train': trainVectorize, 'test': testVectorize},
    'histogram': {'train': trainHistogram, 'test': testHistogram},
    'downsampling': {'train': trainDownsampling, 'test': testDownsampling,
    'another': {'train': trainAnother, 'test': testAnother}
}
}


# Hàm tính khoảng cách
def calculateDistances(featureMethod: str, testVectorIndex: int) -> np.ndarray:
    if featureMethod not in featureData:
        raise ValueError(f"'{featureMethod}' không hợp lệ.")

    # Lấy các mảng đặc trưng tương ứng
    train = featureData[featureMethod]['train']
    test  = featureData[featureMethod]['test']

    # Kiểm tra chỉ mục có hợp lệ không
    if testVectorIndex< 0 or testVectorIndex>= test.shape[0]:
        raise IndexError(f"{testVectorIndex} nằm ngoài phạm vi của tập test (0 đến {test.shape[0] - 1}).")

    # Lấy vector truy vấn (từ tập test)
    queryVector = test[testVectorIndex].reshape(1, -1) # Kích thước (1, D)

    # Tính khoảng cách Euclidean từ queryVector đến tất cả các vector trong train
    # Sử dụng cdist để tối ưu hóa việc tính toán ma trận khoảng cách
    # Kết quả distMatrix có kích thước (1, N)
    distMatrix = cdist(queryVector, train, metric='euclidean')

    # Trả về mảng khoảng cách (loại bỏ chiều đơn, kích thước N)
    return distMatrix.flatten()

# Cách sử dụng:
# Downsampling: calculateDistances('downsampling',index) 
