# Dự án nhận diện kí tự số viết tay

## Dự án này thực hiện nhận diện kí tự số viết tay từ bộ dữ liệu **MNIST** bằng bốn phép trích xuất đặc trưng và thuật toán **k-Nearest Neighbors (k-NN)**. Điểm nổi bật của dự án là việc thử nghiệm và so sánh bốn phương pháp trích xuất đặc trưng khác nhau để tối ưu hóa hiệu suất phân loại.

## 📌Tính năng chính
- Sử dụng bộ dữ liệu chuẩn **MNIST** gồm 60,000 ảnh huấn luyện và 10,000 ảnh kiểm tra.
- Thực hiện bốn kỹ thuật trích xuất đặc trưng:
  1. **Vector hóa**: chuyển đổi ma trận ảnh thành vector phẳng.
  2. **Histogram**: thống kê phân phối cường độ điểm ảnh.
  3. **Downsampling**: giảm độ phân giải để trích xuất các đặc trưng quan trọng nhất.
  4. **Another**: một phép rút đặc trưng do nhóm từ đề xuất, dùng để tối ưu hóa thời gian phân tích.
- Cài đặt thuật toán phân lớp **k-Neareast Neighbors (k-NN)** từ đầu.

## 🛠 Kiến trúc hệ thống
Quy trình xử lý của dự án bao gồm các bước:
1. **Tiền xử lý**: tải và chuẩn hóa dữ liệu MNIST.
2. **Trích xuất đặc trưng**: biến đổi ảnh gốc thành các vector đặc trưng có ý nghĩa.
3. **Đánh giá**: sử dụng k-NN để tìm các mẫu gần nhất trong không gian đặc trưng và đưa ra dự đoán.
4. **Đánh giá**: tính toán độ chính xác và vẽ Confusion matrix (ma trận nhầm lẫn) cho từng phương pháp trích xuất.

## 🚀 Hướng dẫn cài đặt
1. Clone repository
```
git clone https://github.com/LeatuyrBertyk/HandDigits.git
cd HandDigits
```
2. Cài đặt thư viện cần thiết
```
pip install numpy matplotlib scikit-learn pandas
```
3. Chạy chương trình
```
python evaluate.py
```

## 📊 Kết quả thực nghiệm
Bạn có thể xem bằng cách chạy ```evaluate.py ``` và các Confusion matrix trong thư mục ```resultkNN```.
