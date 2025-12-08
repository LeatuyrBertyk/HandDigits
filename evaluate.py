#cài thêm thư viện seaborn
# thư viện matplotlib.pyplot và seaborn để biểu đồ confusion matrix
# import các hàm từ sklearn.metrics để đánh giá 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import từ knnClassifier (nơi đã load dữ liệu và định nghĩa featureData, predictLabel)
from knnClassifier import featureData, predictLabel, testLabels

def evaluateAllMethods(kValue=5):
    """
    Hàm chạy kNN cho tất cả phương pháp rút đặc trưng
    với giá trị k cho trước, trả về kết quả và in báo cáo.
    """
    results = {}
    print(f"--- Bắt đầu đánh giá với k={kValue} ---")

    for featureMethod, data in featureData.items():
        N_test = data['test'].shape[0]
        assignedLabels = np.zeros(N_test, dtype=int)

        print(f"\nĐang xử lý phương pháp: {featureMethod} (Số lượng mẫu: {N_test})")

        for i in range(N_test):
            assignedLabels[i] = predictLabel(featureMethod, i, k=kValue)

        # Tính accuracy và báo cáo
        acc = accuracy_score(testLabels, assignedLabels)
        print(f"-> Accuracy cho {featureMethod}: {acc:.4f}")
        print(classification_report(testLabels, assignedLabels))

        # Vẽ confusion matrix
        cm = confusion_matrix(testLabels, assignedLabels)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {featureMethod} (k={kValue})")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        # Lưu kết quả
        results[featureMethod] = {
            'assignedLabels': assignedLabels,
            'accuracy': acc
        }

    return results


if __name__ == "__main__":
    # Chạy thử với k=5 (hoặc thay bằng k tối ưu từ GridSearch)
    results = evaluateAllMethods(kValue=5)

    print("\n=== Tổng hợp kết quả ===")
    for method, info in results.items():
        print(f"{method}: accuracy = {info['accuracy']:.4f}")
