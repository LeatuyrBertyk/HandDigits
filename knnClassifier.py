import numpy as np
import os
import time
from featureExtract.loadMnist import loadMnist 

dataFolder = 'data'
trainImgs, trainLabels = loadMnist(dataFolder, kind='train')
testImgs, testLabels = loadMnist(dataFolder, kind='t10k')

featureExtractDir = 'resultFeatureExtract' 

def loadFeature(method: str, kind: str) -> np.ndarray:
    filename = f"{kind}{method.capitalize()}.npy"
    filepath = os.path.join(featureExtractDir, filename)
    return np.load(filepath)

trainVectorize = loadFeature('vectorize', 'train')
testVectorize = loadFeature('vectorize', 'test')
trainHistogram = loadFeature('histogram', 'train')
testHistogram = loadFeature('histogram', 'test')
trainDownsampling = loadFeature('downsampling', 'train')
testDownsampling = loadFeature('downsampling', 'test')
trainAnother = loadFeature('another', 'train')
testAnother = loadFeature('another', 'test')

featureData = {
    'vectorize':    {'train': trainVectorize,    'test': testVectorize,    'y_train': trainLabels},
    'histogram':    {'train': trainHistogram,    'test': testHistogram,    'y_train': trainLabels},
    'downsampling': {'train': trainDownsampling, 'test': testDownsampling, 'y_train': trainLabels},
    'another':      {'train': trainAnother,      'test': testAnother,      'y_train': trainLabels}
}

def euclideanDistance(x1,x2) : 
    return np.sqrt(np.sum((x1-x2) ** 2))

def kNNPredictLabel(featureMethod: str, testVectorIndex: int, k) -> np.ndarray:
    # Lấy các mảng đặc trưng tương ứng
    train = featureData[featureMethod]['train']
    test  = featureData[featureMethod]['test']

    queryVector = test[testVectorIndex].flatten()
    
    N_train = train.shape[0]
    distances = np.zeros(N_train) 
    for i in range(N_train):
        distances[i] = euclideanDistance(queryVector, train[i].flatten())

    k_idx = np.argsort(distances)[:k]
    y_train = featureData[featureMethod]['y_train']
    k_labels = y_train[k_idx]
    labels, counts = np.unique(k_labels, return_counts=True)
    majority_label = labels[np.argmax(counts)]

    return majority_label

if __name__ == "__main__": 
    print(f"Kích thước đặc trưng Vectorize (Train): {trainVectorize.shape}")
    print(f"Kích thước đặc trưng Vectorize (Test ): {testVectorize.shape}")
    print(f"Kích thước đặc trưng Histogram (Train): {trainHistogram.shape}")
    print(f"Kích thước đặc trưng Histogram (Test ): {testHistogram.shape}\n")
    kValue = 5 
    results = {}
    output_dir = 'result' 
    
    print(f"Bắt đầu dự đoán nhãn với k={kValue} và lưu vào thư mục '{output_dir}/' \n")
    
    for featureMethod, data in featureData.items():
        N_test = data['test'].shape[0]
        assignedLabels = np.zeros(N_test, dtype=int)
        
        print(f"\n --- Đang xử lý phương pháp: **{featureMethod}** (Số lượng mẫu: {N_test})")
       
        startTime = time.time()
        # Gán nhãn
        for i in range(N_test):
            assignedLabels[i] = kNNPredictLabel(featureMethod, i, k=kValue)
        
        endTime = time.time()
        elapsedTime = endTime-startTime
        print(f"Thời gian chạy: {elapsedTime:.2f} giây\n")
    
        output_filename = f"{featureMethod}_k{kValue}_labels.npy"
        output_filepath = os.path.join(output_dir, output_filename)
        
        np.save(output_filepath, assignedLabels)
        
        results[featureMethod] = {
            'assignedLabels': assignedLabels,
        }
        print(f"-> Đã lưu kết quả vào file: **{output_filepath}**")
