from loadMnist import loadMnist 
import numpy as np

# Ý tưởng: Sau khi thực hiện phép rút đặc trưng bằng phương pháp Histogram, 
#           với mỗi hình, ta thu được một array đặc trưng có kích thước 256. 
#          Tại đây ta sẽ thực hiện thu gọn array đặc trưng này xuống còn 64 phần tử 
#           bằng cách gom nhóm 4 phần tử liền kề nhau lại thành 1 phần tử bằng cách tính trung bình cộng. 

def RutDacTrung(FloatImgs): 
    n,m,p = FloatImgs.shape 
    IntImgs = FloatImgs.astype(int) # Vì mảng ban đầu là kiểu float cần chuyển sang kiểu int
    HistogramArray = np.zeros((n, 256), dtype = int) # Chuẩn bị mảng 
    for i in range(n): 
        HistogramArray[i] = np.bincount(IntImgs[i].ravel(), minlength = 256) # Rút đặc trưng bằng phương pháp Histogram
    HistogramArray = HistogramArray/ (m * p)  # Chuẩn hóa về [0,1] 
    MyHistogramArray = HistogramArray.reshape(n, 64, 4).mean(axis = 2).astype(float) # Gom 4 phần tử liền kề nhau thành 1 phần tử bằng trung bình cộng
    return MyHistogramArray

# # Load 4 mảng numpy (bắt buộc)
dataFolder = 'data' 
TrainImgs, TrainLabels = loadMnist(dataFolder, kind='train')
TestImgs, TestLabels = loadMnist(dataFolder, kind='t10k')
ProcessedTrainImgs = RutDacTrung(TrainImgs) 
ProcessedTestImgs = RutDacTrung(TestImgs) 