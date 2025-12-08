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
import numpy as np
from scipy.spatial.distance import cdist  # Hỗ trợ hàm tính khoảng cách
from scipy import stats

from featureExtract.loadMnist import loadMnist 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from featureExtract.histogram import histogramExtract 
from featureExtract.another import anotherExtract 
from featureExtract.downsampling import downsamplingExtract 

def EuclideanDistance(x1,x2) : 
    return np.sqrt(np.sum((x1-x2) ** 2))

def KNNPredictSingle(Vectors,TestVector, LabelVectors, Kval) : 
    distances = [EuclideanDistance(TestVector,Vector) for Vector in Vectors]
    Kindex = np.argsort(distances)[:Kval] 
    Knearestlabels = LabelVectors[Kindex]

    most_common = stats.mode(Knearestlabels)
    return most_common.mode[0]

def KNNPredict(Vectors, TestVectors, LabelVectors, Kval) : 
    Predictions = [KNNPredictSingle(Vectors,TestVector, LabelVectors, Kval) for TestVector in TestVectors] 
    return np.array(Predictions) 

def CalculateAccuraryScore(PredictionLabels, TrueLabels) : 
    return np.mean(PredictionLabels == TrueLabels) 

def CrossValidation(TrainningVectors, LabelVectors, Kval, cv = 5, rdstate = 42) -> float: 
    kf = KFold(n_splits = cv, shuffle = True, random_state = rdstate) 
    scores = [] 
    for TrainID, ValID in kf.split(TrainningVectors) : 
        TrainFold, ValFold = TrainningVectors[TrainID] , TrainningVectors[ValID] 
        LabelTrainFold, LabelValFold = LabelVectors[TrainID], LabelVectors[ValID] 
        PredictionLabels = KNNPredict(TrainFold,ValFold,LabelTrainFold, Kval) 
        score = CalculateAccuraryScore(PredictionLabels, LabelValFold) 
        scores.append(score) 
    return np.mean(scores) 


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
trainHistogram = np.array(histogramExtract(trainImgs)) 
trainDownsampling = np.array([downsamplingExtract(img) for img in trainImgs])
trainAnother = np.array(anotherExtract(trainImgs)) 

Kvalues = [1,3,5,7,11,13,17,19,23,29,31,37] 
HistogramBestScore, HistogramBestIndex = GridSearch(trainHistogram,trainLabels,Kvalues)
DownsamplingBestScore, DownsamplingBestIndex = GridSearch(trainDownsampling,trainLabels, Kvalues)
AnotherBestScore, AnotherBestIndex = GridSearch(trainAnother,trainLabels,Kvalues)
print(HistogramBestScore, HistogramBestIndex) 
print(DownsamplingBestScore,DownsamplingBestIndex) 
print(AnotherBestScore,AnotherBestIndex) 