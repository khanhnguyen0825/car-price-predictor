# HƯỚNG DẪN THIẾT KẾ SLIDE BÁO CÁO DỰ ÁN: CAR PRICE PREDICTOR

Đây là format slide tiêu chuẩn (11 Slides) được thiết kế chuyên nghiệp để giúp bạn tự tin thuyết trình đơn phương (Solo) hoặc đại diện nhóm.

---

### 🎨 Tông màu định hướng:
- **Style:** Clean & Professional.
- **Màu sắc:** Cam (#FF6B35) & Trắng (phù hợp với Light Mode bạn vừa cập nhật).

---

### 🎞️ CHI TIẾT TỪNG SLIDE

| STT | Tên Slide | Nội dung chính (Chữ ít - Hình nhiều) | Kịch bản nói (Key points) |
| :-- | :--- | :--- | :--- |
| **01** | **Trang bìa** | Tiêu đề: CAR PRICE PREDICTOR. Ảnh 1 chiếc xe sang chảnh mờ ảo. | Giới thiệu tên, đề tài và định hướng dự án là ML thương mại. |
| **02** | **Bối cảnh thị trường** | Thống kê: 15.000+ tin đăng/tháng. Vấn đề: Giá ảo, thiếu công cụ định giá nhanh. | Tại sao chúng ta cần AI? Để thay thế cảm tính bằng dữ liệu. |
| **03** | **Luồng hệ thống** | Sơ đồ 3 khối: Raw Data -> ML Model -> Web App. | Giải thích sự tách biệt giữa pha nghiên cứu (Colab) và pha sản phẩm (Web). |
| **04** | **Dữ liệu & Làm sạch** | Screenshot code Regex xử lý chữ "Tỷ", "Triệu". IQR Outlier chart. | "Rác vào - Rác ra". Giải thích việc lọc từ 15k xuống 8k xe sạch. |
| **05** | **Feature Engineering** | Công thức: `2026 - Year = Age`. Điểm nhấn: `KM_Negative` (Số KM âm). | Giải thích "Trick" đổi dấu KM để mô hình học đúng quy luật hao mòn. |
| **06** | **Thi đấu thuật toán** | Bảng so sánh R2 của 6 Models (Linear, SVR, Random Forest...). | Tại sao Linear thất bại (19%) và Random Forest vô địch (86%). |
| **07** | **Tại sao Random Forest?** | Ảnh minh họa hàng trăm cây quyết định bầu chọn kết quả. | Tính phi tuyến tính của giá xe (Mer cũ vẫn đắt hơn Kia mới). |
| **08** | **AI giải thích điều gì?** | Biểu đồ Feature Importance (Dòng xe chiếm 45% ảnh hưởng). | Tiết lộ "bí mật": Dòng xe cụ thể quan trọng hơn Hãng xe. |
| **09** | **Web App Demo** | Screenshot giao diện Tab 1 (Dự đoán). Chữ to: "LIVE DEMO". | Chuyển sang màn hình trình duyệt để thao tác thực tế. |
| **10** | **Hướng phát triển** | Icon: Camera AI, Real-time crawling. | Dự án chưa hoàn hảo (thiếu dữ liệu xe tai nạn), sẽ nâng cấp sau. |
| **11** | **Q&A & Thank You** | Mã QR dẫn đến Github dự án + Lời cảm ơn. | Sẵn sàng nhận câu hỏi phản biện từ Hội đồng. |

---

### 💡 LỜI KHUYÊN ĐỂ LẤY ĐIỂM "A+"
1. **Show Evidence (Show bằng chứng):** Khi nói đến slide 4-5, hãy mở nhẹ phần comment tiếng Việt trong file `data_loader.py` hoặc notebook Colab đã merge để chứng minh mình làm thật.
2. **Handle Failure (Nói về thất bại):** Đừng ngại nói về việc SVR chỉ đạt 5% độ chính xác. Giảng viên đánh giá cao việc bạn biết *tại sao nó fail* (chưa scale dữ liệu) hơn là một kết quả đẹp giả tạo.
3. **Interactive Demo:** Lúc Demo, hãy hỏi giảng viên: "Thầy muốn em định giá thử chiếc xe đời bao nhiêu?" - Sự tương tác này xóa tan nghi ngờ về việc bạn quay video sẵn.
