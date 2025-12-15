import numpy as np
import os
import time
from featureExtract.loadMnist import loadMnist 
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from knnClassifier import kNNPredictLabel
from knnClassifier import loadFeature

# Load train và test bằng Vectorize
featureExtractDir = 'resultFeatureExtract' 
trainVectorize = loadFeature('vectorize', 'train')
testVectorize = loadFeature('vectorize', 'test')


# kValue = [5,7,9,11,13,15,17,19,21,23]
featureMethod = 'vectorize'


# Note vì đã cho ra kết quả ở resultkNN/resultVectorize

# Cài đặt cấu hình cho việc trả kết quả
outputDir = os.path.join('resultkNN', f'result{featureMethod}') 

print(f"Kích thước đặc trưng Train: {trainVectorize.shape}")
print(f"Kích thước đặc trưng Test : {testVectorize.shape}")

N_test = testVectorize.shape[0]
assignedLabels = np.zeros(N_test, dtype=int)

index=9
# for index in kValue:
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




# dataFolder = 'data'
# baseOutputDir = 'resultkNN' 
# subDir = f'result{featureMethod.capitalize()}'
# 
# _, y_true = loadMnist(dataFolder, kind='t10k')
# totalSample = len(y_true)
# 
# 
# def evaluateVectorize(y_true: np.ndarray, k_val: int):
#     print(f"Đánh giá Vectorize với k = {k_val}")
#     
#     predicted_filename = f"{featureMethod}_k{k_val}_labels.npy"
#     predicted_labels_path = os.path.join(baseOutputDir, subDir, predicted_filename)
#     
#     y_pred = np.load(predicted_labels_path)
#     accuracy = accuracy_score(y_true, y_pred)
#     correctPredictions = np.sum(y_pred == y_true)
#     
#     print(f"   * Số mẫu đoán đúng: {correctPredictions}/{totalSample}")
#     print(f"   * Độ chính xác (Accuracy): {accuracy*100:.2f}%\n")
#     
# 
#     # Đánh giá hiệu quả bằng Confuse matrix 
#     conf_matrix = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(8, 7)) 
#     sns.heatmap(conf_matrix, 
#                 annot=True, 
#                 fmt='d', 
#                 cmap='Blues',
#                 xticklabels=[str(i) for i in range(10)], 
#                 yticklabels=[str(i) for i in range(10)],
#                 linewidths=.5, linecolor='black',
#                 cbar=False)
#     
#     plt.xlabel('Nhãn Dự đoán (Predicted)')
#     plt.ylabel('Nhãn Thực tế (True)')
#     plt.title(f'Confusion Matrix - Vectorize (K={k_val}) - Acc: {accuracy*100:.2f}%')
#     
#     save_dir = os.path.join(baseOutputDir, subDir)
#     save_path = os.path.join(save_dir, f'confusion_matrix_{featureMethod}_k{k_val}.png')
#     
#     plt.savefig(save_path)
#     plt.close()
# 
#     return accuracy
# 
# if __name__ == "__main__": 
#     
#     best_k = None
#     max_accuracy = -1.0
#     
#     for k_val in kValue:
#         current_accuracy = evaluateVectorize(y_true, k_val)
#         
#         if current_accuracy > max_accuracy:
#             max_accuracy = current_accuracy
#             best_k = k_val
# 
#     print(f"\nK tốt nhất (dựa trên tập Test): k = {best_k}")
#     print(f"\nĐộ chính xác cao nhất: {max_accuracy*100:.2f}%")
