# Hướng dẫn chạy kiểm chứng — QSDC v4_4

## 1. Bộ code dùng để làm gì

Bộ code kiểm chứng bằng số các kết quả giải tích của bài, và tạo lại mọi con số và hình được tính bằng
máy (không phải chứng minh):
- Bảng I và Bảng II, gồm cả cột p_opt^(2*) mới thêm vào Bảng I.
- Các con số completeness ở §VI-C (câu ngay sau (69)).
- Ba hình D1–D3 của Phụ lục D (Numerical checks).

Không bước chứng minh nào phụ thuộc vào bộ code này.

## 2. Cài đặt (một lần)

```bash
pip install -r requirements.txt
```
Cần Python ≥ 3.10 với numpy, scipy, pandas, matplotlib. Các phiên bản ghi trong `requirements.txt` là
phiên bản của lần chạy tham chiếu.

## 3. Chạy

Mở terminal tại thư mục `QSDC_v4_4_code` và chạy:
```bash
bash run_everything.sh            # chạy đầy đủ, khoảng 10–12 phút
bash run_everything.sh --quick    # khoảng 1–2 phút, dùng lại kết quả tìm kiếm đối kháng M11 có sẵn
```
Trên Windows (không có bash), chạy lần lượt bốn lệnh:
```bash
cd verification && python run_all.py && cd ..
python appendixD/make_figs.py verification/results appendixD/figures
python text_numbers/check_v44_text_numbers.py verification/results/m6_table1.csv
python compare_results.py
```

## 4. Kết quả cần thấy

| Bước | Lệnh | Kết quả đạt |
|---|---|---|
| 1 | `run_all.py` | Bảng 38 mệnh đề, mỗi dòng PASS kèm số liệu; dòng cuối `38/38 passed in ...s` |
| 2 | `make_figs.py` | `appendixD/figures/figD1_schedule.pdf`, `figD2_uncertainty.pdf`, `figD3_completeness.pdf` và `numbers.json` (mọi số trích trong Phụ lục D) |
| 3 | `check_v44_text_numbers.py` | Cột p_opt^(2*) = 0.276, 0.257, 0.238, 0.117, 0.110, 0.102; Hoeffding trong khoảng 0.04–0.91; đuôi nhị thức chính xác < 1e-13; dòng cuối `PASS` |
| 4 | `compare_results.py` | Mỗi file một dòng `OK`; dòng cuối `All files agree with the reference.` |

## 5. Đối chiếu với bài

Bảng ánh xạ module ↔ phương trình/mệnh đề (đánh số theo v4_4) nằm trong `README.md`. Phụ lục D dùng:
- **D1:** M8.
- **D2:** M2 và M11.
- **D3:** M2 và M11.
- **D4:** M10.

Tài liệu `docs/QSDC_numerical_verification_companion_v3.pdf` mô tả mục tiêu, kịch bản, phương pháp và
lưu đồ của từng module.

## 6. Nếu kết quả khác tham chiếu

Mọi phép ngẫu nhiên dùng seed cố định. Với cùng phiên bản thư viện, kết quả trùng tới làm tròn. Với
phiên bản khác:
- Các đại lượng chính xác (M1, M3–M6, M9, M10) vẫn khớp tới khoảng 1e-10.
- Các đại lượng Monte Carlo (M7, M8) có thể lệch trong phạm vi sai số thống kê, khoảng 1e-3.
- Bộ tối ưu M11 có thể dừng ở điểm khác, nhưng các tiêu chí PASS vẫn phải đạt.

Nếu `compare_results.py` báo `DIFFERS` nhưng độ lệch nằm trong giới hạn trên và `run_all.py` vẫn
báo 38/38 PASS, kết quả vẫn được chấp nhận.
