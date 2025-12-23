# Tìm giá trị K phù hợp cho từng vector đặc trưng bằng phương pháp Grid Search kết hợp với Cross Validation
# Grid Search ở đây giúp tìm kiếm một giá trị K phù hợp bằng phương pháp liệt kê
# Kết quả của Grid Search trả về trong trường hợp này là một mảng 2 chiều
# Giả sử ô (i,j) có giá trị là v. Trong đó: i là một giá trị K đang xét, j là vector đặc trưng được sử dụng, v là độ chính xác
# Ví dụ ô (3, "vectorize") = 1 có tương đương: khi sử dụng mô hình KNN với K = 3 cho vector đặc trưng 
# được rút bằng phương pháp vectorize thì Cross Validation cho ra độ chính xác là 1
# Cross Validation là một phương pháp đánh giá độ tốt của mô hình bằng cách chia tập test thành nhiều phần khác nhau
# rồi luân phiên dùng mỗi phần để kiểm tra các phần còn lại để huấn luyện
# Ví dụ: chi tập test thành 5 phần khác nhau (1,2,3,4,5) sau đó lần lượt cho phần 1 làm phần kiểm tra (2,3,4,5) làm phần huấn luyện
# Cross-Validation chỉ áp dụng lênh tập train
# Output Another: best accuracy: 0.3229, best K: 37
import numpy as np
from scipy.spatial.distance import cdist  # Hỗ trợ hàm tính khoảng cách
from scipy import stats

from featureExtract.loadMnist import loadMnist 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from featureExtract.vectorize import vectorizeExtract
from featureExtract.histogram import histogramExtract 
from featureExtract.another import anotherExtract 
from featureExtract.downsampling import downsamplingExtract 


# Mô hình KNN

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
    
    unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
    prediction = unique_labels[np.argmax(counts)]
    
    return prediction

def KNNPredict(Vectors, TestVectors, LabelVectors, Kval) : 
    Predictions = [PredictLabel(TestVector,Vectors, LabelVectors, Kval) for TestVector in TestVectors] 
    return np.array(Predictions) 

#Cross Validation
def CalculateAccuraryScore(PredictionLabels, TrueLabels) :  # Đánh giá độ chính xác của mô hình KNN 
    return np.mean(PredictionLabels == TrueLabels) 

def CrossValidation(TrainningVectors, LabelVectors, Kval, cv = 5, rdstate = 42) :
    kf = KFold(n_splits = cv, shuffle = True, random_state = rdstate) 
    scores = [] 
    for TrainID, ValID in kf.split(TrainningVectors) : 
        TrainFold, ValFold = TrainningVectors[TrainID] , TrainningVectors[ValID] 
        LabelTrainFold, LabelValFold = LabelVectors[TrainID], LabelVectors[ValID] 
        PredictionLabels = KNNPredict(TrainFold,ValFold,LabelTrainFold, Kval) 
        score = CalculateAccuraryScore(PredictionLabels, LabelValFold) 
        scores.append(score) 
    return np.mean(scores) 

# Grid Search
def GridSearch(FeatureVectors, LabelVectors, Kvalues) : 
    BestScore = -np.inf 
    BestIndex = [] 
    for Kval in Kvalues: 
        MeanScore = CrossValidation(FeatureVectors,LabelVectors,Kval)
        if MeanScore > BestScore : 
            BestScore = MeanScore
            BestIndex = Kval 
    return BestScore, BestIndex

dataFolder = 'data'
trainImgs, trainLabels = loadMnist(dataFolder, kind='train')
testImgs, testLabels = loadMnist(dataFolder, kind='t10k')


# Load các mảng bằng 4 phương pháp rút đặc trưng
trainVectorize = vectorizeExtract(trainImgs)
trainHistogram = np.array(histogramExtract(trainImgs)) 
trainDownsampling = np.array([downsamplingExtract(img) for img in trainImgs])
trainAnother = np.array(anotherExtract(trainImgs)) 

Kvalues =  [7,13,29,37,47,59,79,83,97,103,111]
AnotherBestScore, AnotherBestIndex = GridSearch(trainAnother,trainLabels,Kvalues)
print(AnotherBestScore,AnotherBestIndex) 
Kvalues = [5,7,9,11,13,15,17,19,21,23] 
VectorizeBestScore, VectorizeBestIndex = GridSearch(trainVectorize,trainLabels,Kvalues) 
print(VectorizeBestScore, VectorizeBestIndex)
Kvalues =  [5,7,9,11,13,15,17,19,21,23]
HistogramBestScore, HistogramBestIndex = GridSearch(trainHistogram,trainLabels,Kvalues)
print(HistogramBestScore, HistogramBestIndex)
Kvalues =   [5,7,9,11,13,15,17,19,21,23]
DownsamplingBestScore, DownsamplingBestIndex = GridSearch(trainDownsampling,trainLabels, Kvalues)
print(DownsamplingBestScore,DownsamplingBestIndex) 
