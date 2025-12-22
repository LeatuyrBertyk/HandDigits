import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
import os
from knnClassifier import loadFeature
from featureExtract.loadMnist import loadMnist


def histogramExtract(img):
    hist = np.zeros(256, dtype=float)
    for pixel in img.flatten():
        hist[int(pixel)] += 1
    hist = hist / img.size
    return hist


def PredictLabel(new_vector, train_vectors, train_labels, k):
    """
    Dự đoán nhãn bằng KNN sử dụng cdist để tối ưu tốc độ.
    
    Tham số:
    - new_vector: Vector cần dự đoán (1D array-like).
    - train_vectors: Tập hợp các vector huấn luyện (2D array-like).
    - train_labels: Tập hợp các nhãn tương ứng (1D array-like).
    - k: Số lượng láng giềng.
    """ 
    query = np.atleast_2d(new_vector)
    train_x = np.asarray(train_vectors)
    train_y = np.asarray(train_labels)

    distances = cdist(query, train_x, metric='euclidean')[0]
    k_indices = np.argpartition(distances, k)[:k]
    k_nearest_labels = train_y[k_indices]

    labels, counts = np.unique(k_nearest_labels, return_counts=True)
    return labels[np.argmax(counts)]


def ProcessImageToNumpy(image_path):
    """
    Đọc file PNG, chuyển về ảnh xám, resize về 28x28 và chuyển thành mảng Numpy.
    """
    try:
        # 1. Kiểm tra file có tồn tại không
        if not os.path.exists(image_path):
            print("Lỗi: File không tồn tại!")
            return None
        # 2. Mở ảnh
        with Image.open(image_path) as img:
            # 3. Chuyển đổi sang ảnh xám (L mode: 8-bit pixels, black and white)
            # Bước này quan trọng để đưa từ 3 kênh màu (RGB) về 1 kênh duy nhất
            img_gray = img.convert('L')
            # 4. Thay đổi kích thước về 28x28
            # Sử dụng Resampling.LANCZOS để giữ chất lượng ảnh tốt nhất khi thu nhỏ
            img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)
            # 5. Chuyển sang mảng Numpy
            img_array = np.array(img_resized).astype(np.uint8)
            return img_array

    except Exception as e:
        print(f"Đã xảy ra lỗi khi xử lý ảnh: {e}")
        return None


_, HistogramLabels = loadMnist('data', kind='train')
HistogramTrain = loadFeature('histogram', 'train')

path = input("Nhập đường dẫn file ảnh PNG: ")
vector_img = ProcessImageToNumpy(path)

if vector_img is not None:
    histogram_img = histogramExtract(vector_img)
    prediction = PredictLabel(
        histogram_img,
        HistogramTrain,
        HistogramLabels,
        k=11
    )
    print("Dự đoán chữ số trên ảnh:", prediction)
