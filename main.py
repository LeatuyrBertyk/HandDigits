# Khai báo thư viện
import matplotlib.pyplot as plt
import os
import numpy as np

# Đọc các file dữ liệu Mnist ở dạng binary sang numpy
def loadMnist(path,kind='train'):
	labelsPath=os.path.join(path,'%s-labels.idx1-ubyte' % kind)
	imagesPath=os.path.join(path,'%s-images.idx3-ubyte' % kind)

	with open(labelsPath,'rb') as lbpath:
		lbpath.read(8)
		buffer=lbpath.read()
		labels=np.frombuffer(buffer,dtype=np.uint8)
	with open(imagesPath,'rb') as imgpath:
		imgpath.read(16)
		buffer=imgpath.read()
		images=np.frombuffer(buffer,dtype=np.uint8).reshape(len(labels),28,28).astype(np.float64)
	return images,labels

# Kiểm tra xem dữ liệu đã được chuẩn hóa chưa


# Code xuất sang đồ thị của thầy (tìm trong dữ liệu đã chuẩn hóa hình đầu tiên mỗi kí tự)
# xTrain, yTrain=loadMnist('mnistData/', kind='train')
# print('Row: %d, columns: %d % (xTrain.shape[0],XTrain.shape[1])')
# fig, ax =plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True,)
# ax=ax.flatten()
# for i in range(10):
# 	img=xTrain[yTrain==i][0]
# 	ax[i].imshow(img,cmap='Greys',interpolation='nearest')

# ax[0].set_xticks([])
# ax[0].set_yticks([])
# plt.tight_layout()
# plt.show()

# In ra kích thước các file để kiểm tra xem dữ liệu đã được chuẩn hóa chưa
dataFolder = 'data' 

print("Đang đọc dữ liệu...")
trainImgs, trainLabels = loadMnist(dataFolder, kind='train')
testImgs, testLabels = loadMnist(dataFolder, kind='t10k')

print(f"1. trainImgs shape:   {trainImgs.shape}")   # Mong đợi: (60000, 28, 28)
print(f"2. trainLabels shape: {trainLabels.shape}") # Mong đợi: (60000,)
print(f"3. testImgs shape:    {testImgs.shape}")    # Mong đợi: (10000, 28, 28)
print(f"4. testLabels shape:  {testLabels.shape}")  # Mong đợi: (10000,)

print("Đã chuẩn bị xong 4 tập dữ liệu!")