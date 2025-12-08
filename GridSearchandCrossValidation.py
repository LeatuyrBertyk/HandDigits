# Tìm giá trị K phù hợp cho từng vector đặc trưng bằng phương pháp Grid Search kết hợp với Cross Validation
# Grid Search ở đây giúp tìm kiếm một giá trị K phù hợp bằng phương pháp liệt kê
# Kết quả của Grid Search trả về trong trường hợp này là một mảng 2 chiều
# Giả sử ô (i,j) có giá trị là v. Trong đó: i là một giá trị K đang xét, j là vector đặc trưng được sử dụng, v là độ chính xác
# Ví dụ ô (3, "vectorize") = 1 có tương đương: khi sử dụng mô hình KNN với K = 3 cho vector đặc trưng 
# được rút bằng phương pháp vectorize thì Cross Validation cho ra độ chính xác là 1
# Cross Validation là một phương pháp đánh giá độ tốt của mô hình bằng cách chia tập test thành nhiều phần khác nhau
# rồi luân phiên dùng mỗi phần để kiểm tra các phần còn lại để huấn luyện
# Ví dụ: chi tập test thành 5 phần khác nhau (1,2,3,4,5) sau đó lần lượt cho phần 1 làm phần kiểm tra, (2,3,4,5) làm phần huấn luyện
