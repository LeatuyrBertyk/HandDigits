from 
import numpy as np

# tải dữ liệu MNIST
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.reshape(-1, 28, 28)
y = mnist.target.astype(int)

# trích HOG cho từng ảnh
hog_features = []
for img in X:
    feat = hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), orientations=9)
    hog_features.append(feat)

hog_features = np.array(hog_features)

# chia tập
X_train, X_test, y_train, y_test = train_test_split(hog_features, y, test_size=0.2)

# train SVM
clf = LinearSVC()
clf.fit(X_train, y_train)

# accuracy
print("Accuracy:", clf.score(X_test, y_test))
