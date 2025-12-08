from featureExtract.loadMnist import loadMnist 
import numpy as np

# Load 4 mảng numpy 
dataFolder = 'data' 
trainImgs, trainLabels = loadMnist(dataFolder, kind='train')
testImgs, testLabels = loadMnist(dataFolder, kind='t10k')

# Hàm downsampeImage
def  downsamplingExtract(img, block_size = 4):
    h, w = img.shape # Lấy height(h) và width(w) của img, h = 28, w = 28
    new_h = h // block_size # heigh mới = 4
    new_w = w // block_size # width mới = 4
    small_img = np.zeros((new_h, new_w))  # mảng để lưu ảnh sau khi downsampling

    # Vòng lặp để tạo ra chuyển ma trận 28 * 28 về 7 * 7
    for i in range(new_h):
        for j in range(new_w):
            block = img[i * block_size: (i + 1) * block_size, 
                        j * block_size: (j + 1) * block_size] # lấy block 4 * 4
            small_img[i, j] = np.mean(block) # tạo ra pixel mới 
        
    # Chuẩn hóa về [0,1]
    small_img = small_img / 255.0
    # Chuyển về nhị phân (0/1)
    binary_img = (small_img > 0). astype(int)
    return binary_img.flatten()

# Áp dụng downsampling cho toàn bộ dataset
processedTrainFeatures = np.array([downsamplingExtract(img) for img in trainImgs])
processedTestFeatures = np.array([downsamplingExtract(img) for img in testImgs])

# In kết quả
print("Train features shape:", processedTrainFeatures.shape)
print("Test feature shape:", processedTestFeatures.shape)

# Lưu kết quả để dùng cho các bước tiếp theo
# np.save("train_downsampling.npy", processedTrainFeatures)
# np.save("test_downsampling.npy", processedTrainFeatures)
