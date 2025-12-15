import numpy as np
import os
import time
from featureExtract.loadMnist import loadMnist 
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from knnClassifier import kNNPredictLabel
from knnClassifier import loadFeature



# Note vì đã cho ra kết quả ở resultkNN/resultDownsampling

# # Cài đặt cấu hình cho việc trả kết quả
# outputDir = os.path.join('resultkNN', f'result{featureMethod}') 
# 
# print(f"Kích thước đặc trưng Train: {trainDownsampling.shape}")
# print(f"Kích thước đặc trưng Test : {testDownsampling.shape}")
# 
# N_test = testDownsampling.shape[0]
# assignedLabels = np.zeros(N_test, dtype=int)
# 
# for index in kValue:
#     startTime = time.time()
#         
#     for i in range(N_test):
#         assignedLabels[i] = kNNPredictLabel(featureMethod, i, k=index)
#         
#     endTime = time.time()
#     elapsedTime = endTime - startTime
#         
#     print(f"Thời gian chạy: {elapsedTime:.2f} giây")
#         
#     output_filename = f"{featureMethod}_k{index}_labels.npy"
#     output_filepath = os.path.join(outputDir, output_filename)
#             
#     np.save(output_filepath, assignedLabels)
#     print(f"\n-> Đã lưu mảng nhãn kết quả vào file: {output_filepath}")

# Load train và test bằng another 
featureExtractDir = 'resultFeatureExtract' 
trainDownsampling = loadFeature('another', 'train')
testDownsampling = loadFeature('another', 'test')


kValue = [7, 13, 29, 37, 47, 59, 79, 83, 97, 103, 111, 125, 151, 176, 201]
featureMethod = 'another'





dataFolder = 'data'
baseOutputDir = 'resultkNN' 
subDir = f'result{featureMethod.capitalize()}'

_, y_true = loadMnist(dataFolder, kind='t10k')
totalSample = len(y_true)


def evaluateAnother(y_true: np.ndarray, k_val: int):
    print(f"Đánh giá Another với k = {k_val}")
    
    predicted_filename = f"{featureMethod}_k{k_val}_labels.npy"
    predicted_labels_path = os.path.join(baseOutputDir, subDir, predicted_filename)
    
    y_pred = np.load(predicted_labels_path)
    accuracy = accuracy_score(y_true, y_pred)
    correctPredictions = np.sum(y_pred == y_true)
    
    print(f"   * Số mẫu đoán đúng: {correctPredictions}/{totalSample}")
    print(f"   * Độ chính xác (Accuracy): {accuracy*100:.2f}%\n")
    

    # Đánh giá hiệu quả bằng Confuse matrix 
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 7)) 
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=range(10), yticklabels=range(10),
                linewidths=.5, linecolor='black',
                cbar=False)
    
    plt.xlabel('Nhãn Dự đoán (Predicted)')
    plt.ylabel('Nhãn Thực tế (True)')
    plt.title(f'Confusion Matrix - Another (K={k_val}) - Acc: {accuracy*100:.2f}%')
    
    save_dir = os.path.join(baseOutputDir, subDir)
    save_path = os.path.join(save_dir, f'confusion_matrix_{featureMethod}_k{k_val}.png')
    
    plt.savefig(save_path)
    plt.close()

    return accuracy

if __name__ == "__main__": 
    
# Note vì đã cho ra kết quả ở resultkNN/resultDownsampling

# # Cài đặt cấu hình cho việc trả kết quả
    outputDir = os.path.join('resultkNN', subDir) 
    inputDir = os.path.join('resultFeatureExtract') 
    trainAnother = np.load(os.path.join(inputDir,'trainAnother.npy'))
    testAnother = np.load(os.path.join(inputDir,'testAnother.npy')) 
    print(f"Kích thước đặc trưng Train: {trainAnother.shape}")
    print(f"Kích thước đặc trưng Test : {testAnother.shape}")

    N_test = testAnother.shape[0]
    assignedLabels = np.zeros(N_test, dtype=int)

    for index in kValue:
        startTime = time.time()
            
        for i in range(N_test):
            assignedLabels[i] = kNNPredictLabel(featureMethod, i, k=index)
            
        endTime = time.time()
        elapsedTime = endTime - startTime
            
        print(f"Thời gian chạy: {elapsedTime:.2f} giây")
            
        output_filename = f"{featureMethod}_k{index}_labels.npy"
        output_filepath = os.path.join(outputDir, output_filename)
                
        np.save(output_filepath, assignedLabels)
        print(f"\n-> Đã lưu mảng nhãn kết quả vào file: {output_filepath}")

    best_k = None
    max_accuracy = -1.0
    
    for k_val in kValue:
        current_accuracy = evaluateAnother(y_true, k_val)
        
        if current_accuracy > max_accuracy:
            max_accuracy = current_accuracy
            best_k = k_val

    print(f"\nK tốt nhất (dựa trên tập Test): k = {best_k}")
    print(f"\nĐộ chính xác cao nhất: {max_accuracy*100:.2f}%")
