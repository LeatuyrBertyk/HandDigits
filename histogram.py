from loadMnist import loadMnist
from skimage.feature import hog
from sklearn.svm import LinearSVC
import numpy as np

# 1. Load dữ liệu MNIST từ file binary
X_train, y_train = loadMnist("mnist", kind='train')
X_test, y_test   = loadMnist("mnist", kind='t10k')

# 2. Extract HOG cho training images
hog_train = []
for img in X_train:
    feat = hog(img,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               orientations=9)
    hog_train.append(feat)

hog_train = np.array(hog_train)

# 3. Extract HOG cho test images
hog_test = []
for img in X_test:
    feat = hog(img,
               pixels_per_cell=(8, 8),
               cells_per_block=(2, 2),
               orientations=9)
    hog_test.append(feat)

hog_test = np.array(hog_test)

# 4. Train SVM classifier
clf = LinearSVC()
clf.fit(hog_train, y_train)

# 5. Accuracy
print("Accuracy:", clf.score(hog_test, y_test))
