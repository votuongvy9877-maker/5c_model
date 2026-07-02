# Ứng dụng Dự báo Rủi ro Tín dụng (PD)

Ứng dụng Streamlit chuyển thể từ notebook `PPĐL_RRTD.ipynb`. Ứng dụng huấn luyện một mô hình
**Logistic Regression** để dự báo khả năng rủi ro (biến `PD`: 0 = không rủi ro, 1 = có rủi ro)
dựa trên 24 biến khảo sát thuộc 5 nhóm:

- **TC** (TC1–TC5)
- **NL** (NL1–NL4)
- **DK** (DK1–DK5)
- **V** (V1–V6)
- **TS** (TS1–TS4)

Mô hình được huấn luyện trực tiếp trên các biến gốc (notebook không sử dụng bước chuẩn hóa/scaler
nào trước khi đưa vào `LogisticRegression`), do đó ứng dụng tái hiện đúng pipeline này.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
streamlit run app.py
```

## Cấu trúc dữ liệu đầu vào

File CSV tải lên cần chứa tối thiểu các cột sau:

| Nhóm biến | Các cột |
|---|---|
| Trách nhiệm (TC) | TC1, TC2, TC3, TC4, TC5 |
| Năng lực (NL) | NL1, NL2, NL3, NL4 |
| Điều kiện (DK) | DK1, DK2, DK3, DK4, DK5 |
| Vốn (V) | V1, V2, V3, V4, V5, V6 |
| Tài sản đảm bảo (TS) | TS1, TS2, TS3, TS4 |
| Mục tiêu | PD (0 hoặc 1) |

Các cột khác (ví dụ `Dấu thời gian`, `NN`) có trong file dữ liệu gốc nhưng **không** được dùng
làm biến đầu vào mô hình, đúng theo notebook gốc.

## Mô tả các tab

1. **🗂️ Tổng quan dữ liệu** — Kích thước dữ liệu, xem nhanh dữ liệu thô, thống kê mô tả cho
   24 biến đầu vào và biến mục tiêu PD.
2. **📈 Trực quan hóa dữ liệu** — Phân phối biến mục tiêu PD, và phân phối các biến đầu vào
   (mặc định 4 biến: TC1, NL1, DK1, V1; có thể chọn lại qua ô multiselect vì có tới 24 biến).
3. **🎯 Kết quả huấn luyện & kiểm định mô hình** — Accuracy, Precision, Recall, F1-score,
   ROC-AUC, ma trận nhầm lẫn, đường cong ROC, và báo cáo phân loại chi tiết. Chỉ hiển thị sau
   khi bấm nút huấn luyện ở thanh bên trái.
4. **🔮 Sử dụng mô hình** — Hai chế độ:
   - *Nhập trực tiếp*: nhập giá trị cho từng biến đầu vào để dự báo một trường hợp.
   - *Tải file hàng loạt*: tải lên file CSV có đúng các cột biến đầu vào để dự báo hàng loạt và
     tải kết quả về dưới dạng CSV.

## Ghi chú kỹ thuật

- Notebook gốc gọi `LogisticRegression()` không truyền tham số (dùng toàn bộ giá trị mặc định
  của scikit-learn). Ứng dụng bổ sung các widget cho `C`, `max_iter`, `solver` để người dùng có
  thể tinh chỉnh, với giá trị mặc định đúng bằng mặc định của scikit-learn (`C=1.0`,
  `max_iter=100`, `solver="lbfgs"`).
- `test_size=0.10` và `random_state=23` được lấy đúng theo notebook gốc (`train_test_split`).
- Mô hình chỉ được huấn luyện lại khi người dùng bấm nút **"🚀 Huấn luyện mô hình"** ở thanh
  bên trái; kết quả được lưu trong `st.session_state` để các tab khác dùng lại mà không cần
  huấn luyện lại khi chuyển tab.
- Khuyến nghị dùng Streamlit phiên bản mới (≥ 1.38) để đảm bảo tương thích đầy đủ với các
  thành phần bố cục (`st.container(height=...)`, `st.tabs`, `st.form`).
