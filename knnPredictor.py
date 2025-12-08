import numpy as np
from knnClassifier import calculateDistances
from knnClassifier import featureData
from featureExtract.vectorize import testFeatures 



def predictLabel(featureMethod, testIndex, k):
    # 1. Lấy khoảng cách
    distances = calculateDistances(featureMethod, testIndex)

    # 2. Lấy k chỉ số nhỏ nhất
    k_idx = np.argsort(distances)[:k]

    # 3. Lấy nhãn tương ứng
    y_train = featureData[featureMethod]['y_train']
    k_labels = y_train[k_idx]

    # 4. Majority vote
    labels, counts = np.unique(k_labels, return_counts=True)
    majority_label = labels[np.argmax(counts)]

    return majority_label
# GÁN NHÃN CHO TOÀN BỘ TẬP TEST
# Số lượng mẫu test
N = testFeatures.shape[0]
# Tạo mảng rỗng để chứa nhãn dự đoán của từng ảnh test
# Gọi hàm predictLabel để dự đoán nhãn cho mẫu test thứ i
    # "vector" là tên phương pháp rút đặc trưng bạn dùng
    # k=5 nghĩa là dùng 5 hàng xóm gần nhất
assignedLabels = np.zeros(N, dtype=int)

for i in range(N):
    assignedLabels[i] = predictLabel("vector", i, k=5)

# assignedLabels lúc này chứa nhãn dự đoán của toàn bộ tập test
print("Đã gán nhãn cho toàn bộ tập test.")
print("assignedLabels shape =", assignedLabels.shape)
