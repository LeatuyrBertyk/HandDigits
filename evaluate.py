import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report
from featureExtract.loadMnist import loadMnist

# 1. Load nhan that cua tap test
dataFolder = 'data'
_, testLabels = loadMnist(dataFolder, kind='t10k')

# 2. Thu muc chua cac file ket qua
result_dir = 'result'
feature_methods = ['vectorize', 'histogram', 'downsampling', 'another']
kValue = 5

# 3. Ham danh gia cho tat ca phuong phap
def evaluateAllMethods():
    results = {}
    for method in feature_methods:
        filename = f"{method}_k{kValue}_labels.npy"
        filepath = os.path.join(result_dir, filename)

        if not os.path.exists(filepath):
            print(f"Khong tim thay file ket qua cho {method}")
            continue

        # Load nhan du doan tu file .npy
        assignedLabels = np.load(filepath)

        # Tinh accuracy
        acc = accuracy_score(testLabels, assignedLabels)
        print("\n======================================================")
        print(f"Phuong phap: {method}")
        print(f"Accuracy (k={kValue}): {acc:.4f}")
        print("Classification Report:")
        print(classification_report(testLabels, assignedLabels))

        results[method] = {
            'assignedLabels': assignedLabels,
            'accuracy': acc
        }

    return results

# 4. Chay chuong trinh
if __name__ == "__main__":
    results = evaluateAllMethods()
    print("\n=== Tong hop ket qua ===")
    for method, info in results.items():
        print(f"{method}: accuracy = {info['accuracy']:.4f}")


# ===========================
# Giai thich cac thong so chuong trinh in ra
# 
# Accuracy: do chinh xac tong the, la ti le so mau du doan dung tren tong so mau test
# Cong thuc: Accuracy = so du doan dung / tong so mau test
#
# Classification Report: bang chi tiet cho tung lop (chu so 0-9)
# - Precision: ti le du doan dung trong so tat ca mau ma mo hinh du doan la lop do
#   Vi du: mo hinh du doan 100 anh la so 3, trong do 90 anh thuc su la so 3 -> Precision = 0.90
# - Recall: ti le du doan dung trong so tat ca mau thuc su thuoc lop do
#   Vi du: co 120 anh thuc su la so 3, mo hinh du doan dung 90 anh -> Recall = 0.75
# - F1-score: trung binh hai hoa giua Precision va Recall
#   Cong thuc: F1 = 2 * (Precision * Recall) / (Precision + Recall)
# - Support: so luong mau thuc su thuoc lop do trong tap test
#
# Tong hop Accuracy cuoi cung: so sanh do chinh xac giua cac phuong phap rut dac trung
# Vi du:
# vectorize: accuracy = 0.92
# histogram: accuracy = 0.88
# downsampling: accuracy = 0.85
# another: accuracy = 0.90
# ===========================

