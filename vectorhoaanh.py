import numpy as np
import os

# ============================================================
# ============================================================

# Kiểm tra file MNIST
required_files = [
    'vectorhoaanh/data/train-images.idx3-ubyte',
    'vectorhoaanh/data/train-labels.idx1-ubyte',
    'vectorhoaanh/data/t10k-images.idx3-ubyte',
    'vectorhoaanh/data/t10k-labels.idx1-ubyte'
]

for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Không tìm thấy file: {f}")
print("✓ Đã tìm thấy đầy đủ 4 file MNIST!\n")

# Hàm load dữ liệu
def load_img(f):
    data = open(f, 'rb').read()
    _, n, h, w = np.frombuffer(data[:16], '>i4')
    return np.frombuffer(data[16:], 'u1').reshape(n, h, w).astype('f4') / 255

def load_lbl(f):
    data = open(f, 'rb').read()
    _, n = np.frombuffer(data[:8], '>i4')
    return np.frombuffer(data[8:], 'u1')

# Load dữ liệu
X_train = load_img('vectorhoaanh/data/train-images.idx3-ubyte')
y_train = load_lbl('vectorhoaanh/data/train-labels.idx1-ubyte')
X_test  = load_img('vectorhoaanh/data/t10k-images.idx3-ubyte')
y_test  = load_lbl('vectorhoaanh/data/t10k-labels.idx1-ubyte')

print(f"Dữ liệu gốc: Train {X_train.shape}, Test {X_test.shape}")

# ============================================================
# VECTOR HÓA ẢNH (Flatten 28x28 → 784 chiều)
# ============================================================
X_train_vec = X_train.reshape(-1, 784)
X_test_vec = X_test.reshape(-1, 784)

print(f"\nSau vector hóa: Train {X_train_vec.shape}, Test {X_test_vec.shape}")
print(f"→ Mỗi ảnh 28x28 được chuyển thành vector 784 chiều\n")

# ============================================================
# NHỊ PHÂN HÓA (0 và 1)
# ============================================================
X_train_bin = (X_train_vec >= 0.5).astype('int')
X_test_bin = (X_test_vec >= 0.5).astype('int')

print(f"Sau nhị phân hóa: Train {X_train_bin.shape}, Test {X_test_bin.shape}")
print(f"→ Giá trị >= 0.5 → 1, giá trị < 0.5 → 0\n")

# ============================================================
# IN VÍ DỤ VECTOR ĐẶC TRƯNG
# ============================================================
print("="*60)
print("VÍ DỤ VECTOR ĐẶC TRƯNG (NHỊ PHÂN)")
print("="*60)

# Ví dụ 1
print(f"\n[Ví dụ 1] Ảnh đầu tiên - Label: {y_train[0]}")
print(f"Vector 784 chiều đầy đủ (nhị phân 0/1):")
print(X_train_bin[0])

# Ví dụ 2
print(f"\n[Ví dụ 2] Ảnh thứ hai - Label: {y_train[1]}")
print(f"Vector 784 chiều đầy đủ (nhị phân 0/1):")
print(X_train_bin[1])

print("\n(Mỗi vector có 784 giá trị: 0 hoặc 1)")
print()

# ============================================================
# KẾT QUẢ
# ============================================================
print("="*60)
print("="*60)
print("Phương pháp        : Vector hóa ảnh")
print("Số chiều đặc trưng : 784 (từ ảnh 28x28)")
print("Dữ liệu train      : 60,000 mẫu")
print("Dữ liệu test       : 10,000 mẫu")
print("="*60)
print("\nĐã hoàn thành rút trích đặc trưng bằng vector hóa ảnh")

