import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from featureExtract.loadMnist import loadMnist 
from knnClassifier import kNNPredictLabel, loadFeature


# Hàm tính độ chính xác và Confusion matrix
def evaluateMethod(featureMethod, kValues, yTrue, totalSample, baseOutputDir):
    # Cài đặt cấu hình cho thư mục kết quả
    subDir = f"result{featureMethod.capitalize()}"
    outputDir = os.path.join(baseOutputDir, subDir)
    os.makedirs(outputDir, exist_ok=True)

    # Load tập train và test đã rút đặc trưng
    trainData = loadFeature(featureMethod, 'train')
    testData = loadFeature(featureMethod, 'test')
    
    print(f"--- Đang xử lí phương pháp: {featureMethod.capitalize()} ---")
    print(f"Kích thước đặc trưng Train: {trainData.shape}")
    print(f"Kích thước đặc trưng Test : {testData.shape}")

    nTest = testData.shape[0]
    bestK = None
    maxAccuracy = -1.0

    for kVal in kValues:
        predictedFilename = f"{featureMethod}_k{kVal}_labels.npy"
        predictedFilePath = os.path.join(outputDir, predictedFilename)
        
        # Xử lí dựa trên mảng numpy kết quả
        if os.path.exists(predictedFilePath):
            yPred = np.load(predictedFilePath)
        else: # Thực hiện load sang mảng numpy nếu chưa tồn tại
            print(f" * k={kVal}: Đang dự đoán...")
            startTime = time.time()
            yPred = np.zeros(nTest, dtype=int)
            for i in range(nTest):
                yPred[i] = kNNPredictLabel(featureMethod, i, k=kVal)
            
            endTime = time.time()
            print(f" * Thời gian thực hiện: {endTime - startTime:.2f} giây")
            np.save(predictedFilePath, yPred)
        
        # Công việc 1: Tính độ chính xac
        accuracy = accuracy_score(yTrue, yPred)
        correctPredictions = np.sum(yPred == yTrue)
        print(f"   Số mẫu đúng: {correctPredictions}/{totalSample} | Độ chính xác: {accuracy*100:.2f}%")

        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            bestK = kVal
        
        # Công việc 2: Confusion matrix
        confMatrix = confusion_matrix(yTrue, yPred)
        plt.figure(figsize=(8, 7))
        sns.heatmap(confMatrix, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=[str(i) for i in range(10)], 
                    yticklabels=[str(i) for i in range(10)],
                    linewidths=.5, linecolor='black',
                    cbar=False)
        
        plt.xlabel('Nhãn Dự đoán (Predicted)')
        plt.ylabel('Nhãn Thực tế (True)')
        plt.title(f'Confusion Matrix - {featureMethod.capitalize()} (K={kVal}) - Acc: {accuracy*100:.2f}%')
        
        imageSavePath = os.path.join(outputDir, f'confusion_matrix_{featureMethod}_k{kVal}.png')
        plt.savefig(imageSavePath)
        plt.close()

    print(f"\n * Kết quả tốt nhất cho {featureMethod.capitalize()}: k = {bestK} với {maxAccuracy*100:.2f}%\n")
    return bestK, maxAccuracy

if __name__ == "__main__":
    dataFolder = 'data'
    baseOutputDir = 'resultkNN'
    
    _, yTrue = loadMnist(dataFolder, kind='t10k')
    totalSample = len(yTrue)

    methodsConfig = {
        'vectorize': [5,7,9,11,13,15,17,19,21,23],
        'histogram': [5,7,9,11,13,15,17,19,21,23],
        'downsampling': [5,7,9,11,13,15,17,19,21,23],
        'another': [7,13,29,37,47,59,79,83,97,103,111]
    }

    finalResults = {}

    for method, kList in methodsConfig.items():
        bestK, maxAcc = evaluateMethod(method, kList, yTrue, totalSample, baseOutputDir)
        finalResults[method] = {"bestK": bestK, "accuracy": maxAcc}

    print("BẢNG TỔNG KẾT ĐỘ CHÍNH XÁC")
    print(f"   {'Phương pháp':<15} | {'K tốt nhất':<10} | {'Độ chính xác':<15}")
    for method, res in finalResults.items():
        print(f"   {method.capitalize():<15} | {res['bestK']:<10} | {res['accuracy']*100:.2f}%")
