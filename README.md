# Video Moderation — Mô tả & Giải thích

## Tổng quan
- Ứng dụng này cung cấp API kiểm duyệt video đơn giản bằng FastAPI, lấy mẫu khung hình, phát hiện khuôn mặt, ước lượng nhóm tuổi và phát hiện vật khả nghi (vũ khí) để đánh giá an toàn nội dung.
- Các mã nguồn chính: [main.py](main.py), mô hình vũ khí: [best.pt](best.pt), mô hình tuổi: [age_group_model_final.pth](age_group_model_final.pth), phụ thuộc: [requirements.txt](requirements.txt).

## Luồng xử lý chính (theo `main.py`)
1. Nhận file `.mp4` qua endpoint `/moderate-video`.
2. Trích mẫu khung hình theo `FPS_SAMPLE` (hiện là 0.5s -> ~2 FPS).
3. Dùng bộ phát hiện mặt Haar (`cv2.CascadeClassifier`) để khoá cửa (face gate).
4. Với khung hình có mặt ổn định (ít nhất `MIN_CONSECUTIVE_FACE_FRAMES` khung liên tiếp), chạy mô hình tuổi (`resnet18`) để xác định trẻ em.
5. Song song, chạy YOLO (mô hình `best.pt`) để phát hiện các lớp vũ khí với ngưỡng cấu hình/độ tin cậy (`YOLO_CONF_THRESHOLD`, `WEAPON_CONFIDENCE_THRESHOLD`).
6. Áp quy tắc hậu xử lý: nếu tỉ lệ khung có trẻ em lớn hơn `CHILD_RATIO_THRESHOLD` thì đánh dấu `child` và bỏ qua báo vũ khí (child override). Nếu vũ khí có confidence >= `WEAPON_BAN_THRESHOLD` thì báo `weapon`.

## Tại sao kiểm duyệt có thể sai / không phát hiện / nhầm lẫn (child ↔ weapon)
Những nguyên nhân chính dẫn tới sai lệch và nhầm lẫn:

- Dữ liệu huấn luyện không đầy đủ hoặc mất cân bằng:
  - Thiếu ví dụ thực tế (góc quay, ánh sáng, che khuất, đồ vật tương tự vũ khí), làm mô hình YOLO dễ nhầm lẫn.
  - Mô hình phân loại tuổi thiếu ví dụ đa dạng (da, góc mặt, che mặt), dẫn tới dự đoán không ổn định.

- Vấn đề sampling khung hình và mất thông tin temporal:
  - Lấy mẫu 2 FPS có thể bỏ qua khoảnh khắc chứa vũ khí hoặc khuôn mặt rõ rệt (short event).
  - Mô hình hiện xử lý khung đơn (frame-level) nên không tận dụng động lực học video (motion cues) để phân biệt.

- Chất lượng video và tiền xử lý:
  - Nén, độ phân giải thấp, rung, ánh sáng kém làm giảm độ chính xác cả face detection lẫn object detection.

- Thiết kế ngưỡng và hậu xử lý thô:
  - Ngưỡng confidence tĩnh có thể gây false negatives (ngưỡng quá cao) hoặc false positives (ngưỡng quá thấp).
  - Quy tắc child-override (bỏ qua báo vũ khí khi child detected) là heuristic: nó giảm false positives nhưng cũng có thể che lấp trường hợp có trẻ và vũ khí thật sự.

- Sai khớp không gian giữa hộp vũ khí và mặt:
  - Hiện tại chưa có bước đối sánh không gian giữa vị trí mặt và vị trí vật (spatial correlation). Một vật gần tay có thể không thuộc cùng người có mặt rõ rệt.

- Nhãn không chính xác / nhiễu trong dữ liệu:
  - Nhãn training bị lỗi (label noise) khiến mô hình học sai.

Tác động thực tế: những nguyên nhân trên gây ra ba dạng lỗi phổ biến — false negative (không nhận ra vũ khí), false positive (nhầm vật vô hại thành vũ khí), và lỗi phân lớp tuổi (nhầm trẻ / người lớn).

## Hướng khắc phục & cải tiến trong tương lai
Dưới đây là các bước thực tế bạn có thể trình bày với nhà tuyển dụng để chứng tỏ bạn hiểu vấn đề và có kế hoạch cải thiện:

- Cải thiện dữ liệu và ghi nhãn
  - Thu thập thêm video đa dạng (góc quay, ánh sáng, occlusion, nền khác nhau).
  - Chuẩn hoá hướng dẫn ghi nhãn, kiểm định chất lượng nhãn (double annotation + adjudication).
  - Tạo bộ test giữ riêng (holdout) với các kịch bản khó để đánh giá thực tế.

- Mô hình hoá temporal và không gian
  - Chuyển từ frame-level sang video-level models: 3D CNNs, I3D, hoặc Video Transformer để nắm bắt thông tin động.
  - Dùng object-tracking để nối các hộp vũ khí qua khung (tracklets) và xác định xem vật có di chuyển cùng một người hay không.
  - Không gian hóa: đối sánh bounding box người/mặt với hộp vũ khí để xác định mối quan hệ (ví dụ: vũ khí ở gần tay của người).

- Cải tiến pipeline phát hiện tuổi
  - Dùng detector khuôn mặt hiện đại (MTCNN, RetinaFace) thay vì Haar để bền vững hơn với góc & kích thước.
  - Huấn luyện lại hoặc fine-tune mô hình tuổi với nhiều ví dụ trẻ/emphasis trên edge-cases.

- Ensemble và hai giai đoạn xác nhận
  - Dùng ensemble nhiều mô hình YOLO/SSD với threshold voting để giảm false positives.
  - Thêm bước xác nhận thứ hai: crop vùng nghi ngờ và feed vào classifier chuyên biệt (two-stage verification).

- Ngưỡng động và hiệu chỉnh (calibration)
  - Dùng calibration (Platt scaling, isotonic) hoặc tối ưu threshold dựa trên metric thực tế (precision/recall trade-off) theo từng lớp.
  - Áp dụng ngưỡng khác nhau theo bối cảnh (ví dụ video có trẻ em => thận trọng hơn).

- Hệ thống người can thiệp & active learning
  - Khi model không tự tin (confidence khoảng giữa), đẩy tác vụ cho human reviewer.
  - Lưu các mẫu gây nhầm để đưa vào vòng huấn luyện tiếp theo (active learning).

- Augmentation & dữ liệu tổng hợp
  - Dùng data augmentation (blur, noise, compression) để mô phỏng điều kiện thực tế.
  - Sinh dữ liệu tổng hợp (rendered scenes, synthetic hands holding objects) để tăng đa dạng classes vũ khí.

- Giám sát, logging, và kiểm thử
  - Ghi log chi tiết (timestep, confidence, crops) để debug nguyên nhân sai.
  - Thiết lập metric theo class và báo cáo false positive/negative rate theo kịch bản.

## Những gì tôi đã làm trong dự án này
- Tự train 2 model best.pt (phát hiện súng đạn) và age_group_model_final.pth(phát hiện tuổi => trẻ em)
- Xây dựng API kiểm duyệt video bằng FastAPI (`/moderate-video`).
- Thiết lập pipeline: frame sampling, face gating, age classification (ResNet-18), YOLO weapon detection.
- Hiện thực các heuristic an toàn: child ratio, consecutive face frames, child override để giảm nhạy cảm false positives khi trẻ em xuất hiện.
- Tinh chỉnh các ngưỡng (`YOLO_CONF_THRESHOLD`, `WEAPON_CONFIDENCE_THRESHOLD`, `WEAPON_BAN_THRESHOLD`) để cân bằng precision/recall trong môi trường thử nghiệm.
- Thêm logging/giải pháp tạm thời (ví dụ crop/skip logic trong code) để dễ debug và thử nghiệm các thay đổi threshold.

## Cách chạy (nhanh)
1. Cài các phụ thuộc:

```bash
pip install -r requirements.txt
```

2. Chạy server FastAPI bằng `uvicorn` (từ thư mục chứa `main.py`):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

3. Gửi POST tới `/moderate-video` với một file `.mp4` (multipart/form-data, key `file`).

Ví dụ dùng `curl`:

```bash
curl -F "file=@sample.mp4" http://127.0.0.1:8000/moderate-video
```


