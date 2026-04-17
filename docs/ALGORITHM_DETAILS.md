# Chi Tiết 6 Thuật Toán Machine Learning Trong Dự Án

Dưới đây là phân tích ưu/nhược điểm và kết quả thực tế của 6 mô hình được thử nghiệm trong hệ thống dự báo giá xe.

---

| Thuật toán | Loại | Ưu điểm | Nhược điểm | Kết quả trong dự án |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Tuyến tính | Đơn giản, cực nhanh, dễ giải thích. | Chỉ học được đường thẳng, không bắt được quy luật phức tạp. | **Kém (10.6%)**. Không bắt được sự sụt giảm giá phi tuyến tính. |
| **Ridge Regression** | Tuyến tính | Chống Overfitting tốt hơn Linear nhờ hình phạt L2. | Vẫn bị giới hạn bởi tư duy đường thẳng. | **Kém (10.6%)**. Kết quả tương đương Linear. |
| **Lasso Regression** | Tuyến tính | Có khả năng loại bỏ các đặc trưng thừa (L1). | Có thể loại bỏ nhầm cả các đặc trưng quan trọng. | **Kém (10.6%)**. |
| **SVR (RBF Kernel)** | Phi tuyến | Rất mạnh với dữ liệu phức tạp, không gian nhiều chiều. | Rất nhạy cảm với thang đo (Scaling), chậm khi dữ liệu lớn. | **Thất bại (-4%)**. Do chênh lệch lớn giữa KM và các biến khác. |
| **Gradient Boosting** | Ensemble | Độ chính xác tiềm năng rất cao, học từ sai sót. | Dễ bị Overfitting, khó tinh chỉnh tham số. | **Khá (71.7%)**. Tốt nhưng kém ổn định hơn Random Forest. |
| **Random Forest** | Ensemble | **Ổn định, không cần chuẩn hóa dữ liệu, bắt tương tác biến cực tốt.** | Cần nhiều bộ nhớ hơn để lưu trữ hàng trăm cây. | **🏆 Tốt nhất (86.5%)**. Độ tin cậy cao nhất cho thị trường VN. |

---

## Phân tích sâu về sự khác biệt giữa các nhóm

### 1. Nhóm Tuyến tính (Linear/Ridge/Lasso)
Các mô hình này cố gắng vẽ một đường thẳng để dự đoán giá. Tuy nhiên, giá xe cũ phụ thuộc vào sự kết hợp giữa Hãng xe và Năm sản xuất. 
*   *Lỗi tiêu biểu:* Nó có thể nghĩ rằng xe cứ cũ đi 1 năm thì giá giảm 50 triệu đồng cho **tất cả mọi loại xe**. Điều này hoàn toàn sai vì một chiếc Mercedes cũ giá vẫn có thể cao hơn chiếc Morning mới.

### 2. Nhóm SVR
SVR giống như một vận động viên giỏi nhưng "kén chọn". Nó yêu cầu mọi số liệu đầu vào phải có cùng kích thước (ví dụ từ 0 đến 1). Vì chúng ta đưa vào số KM là 100.000 và Năm là 2020 mà không chuẩn hóa, SVR bị "choáng" và không thể tìm ra quy luật.

### 3. Nhóm Ensemble (Gradient Boosting & Random Forest)
Đây là cách tiếp cận "Sức mạnh tập thể". Thay vì dùng 1 mô hình phức tạp, chúng dùng hàng trăm mô hình đơn giản (Decision Trees) kết hợp lại.
*   **Gradient Boosting** giống như một học sinh giỏi nhưng hay bị áp lực (dễ học thuộc lòng dữ liệu nhiễu).
*   **Random Forest** giống như một nhóm 100 người bình thường cùng thảo luận. Ý kiến trung bình của họ luôn ổn định và ít khi bị sai số nghiêm trọng. Đó là lý do tại sao nó chiến thắng trong dự án này.
