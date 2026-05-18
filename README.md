# NeuroDetect v2.0 - Hệ Thống Phân Loại Sức Khỏe Tinh Thần

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" />
</div>

## Tổng Quan

**NeuroDetect v2.0** là dự án xử lý ngôn ngữ tự nhiên (NLP) toàn diện được xây dựng cho môn học CS221. Mục tiêu chính của dự án là phân loại chính xác các đoạn văn bản trên mạng xã hội thành 4 nhóm sức khỏe tinh thần: **Normal (Bình thường), Anxiety (Lo âu), Depression (Trầm cảm) và Suicidal (Tự tử)**.

Chúng tôi đã huấn luyện, đánh giá và triển khai 5 mô hình khác nhau, từ các thuật toán Machine Learning truyền thống đến các kiến trúc Deep Learning tiên tiến nhất:
1. **BERT (Fine-tuned)** - 84.74% F1-Score
2. **BiLSTM + GloVe** - 82.99% F1-Score
3. **TextCNN** - 81.20% F1-Score
4. **LightGBM** - 78.47% F1-Score
5. **SVM (Baseline)** - 77.91% F1-Score

Dự án đi kèm với một **Giao diện Dashboard phong cách Neo-brutalism** được kết nối trực tiếp tới backend chạy bằng GPU trên Kaggle thông qua `ngrok`, cho phép dự đoán theo thời gian thực và so sánh hiệu năng giữa các mô hình.

---

## Hướng Dẫn Chạy Bản Demo

Để chạy giao diện tương tác, bạn cần khởi động backend API trên Kaggle và frontend ở máy tính cá nhân.

### Bước 1: Khởi Động Kaggle Backend API
1. Truy cập vào Kaggle Notebook: [CS221 Demo Backend](https://www.kaggle.com/code/thaidat733/cs221-demo)
2. Đảm bảo bạn đã bật **GPU (T4 x2 hoặc P100)** và **Internet access** trong phần cài đặt của Kaggle session.
3. Bấm **"Run All"** để chạy toàn bộ notebook.
4. Cuộn xuống cuối notebook. Server FastAPI sẽ được khởi động thông qua `pyngrok`.
5. Copy đường dẫn **Ngrok Public URL** vừa được tạo ra (ví dụ: `https://xxxx-xx-xx-xx.ngrok-free.dev`).

### Bước 2: Khởi Động Local Frontend Dashboard
1. Clone repository này về máy tính của bạn:
   ```bash
   git clone https://github.com/your-username/CS221.git
   cd CS221/Demo
   ```
2. Khởi động HTTP server mặc định của Python:
   ```bash
   python -m http.server 8000
   ```
3. Mở trình duyệt web và truy cập vào địa chỉ: `http://localhost:8000`

### Bước 3: Kết Nối Frontend Với Backend
1. Tại giao diện NeuroDetect Dashboard, bấm vào tab **SETTINGS** ở thanh menu bên trái.
2. Dán đường dẫn **Ngrok Public URL** (vừa copy ở Bước 1) vào ô **API ENDPOINT**. Đảm bảo rằng bạn có thêm hậu tố `/api/predict` vào cuối đường dẫn (ví dụ: `https://xxxx-xx-xx-xx.ngrok-free.dev/api/predict`).
3. Bấm nút **"SAVE CONFIGURATION"**.
4. Chuyển sang tab **TEXT ANALYSIS** hoặc **COMPARISON** để bắt đầu phân tích văn bản!

---

## Cấu Trúc Thư Mục

```text
CS221/
│
├── Data/                   # Dữ liệu gốc và dữ liệu đã qua làm sạch (train/test splits)
│   ├── Clean/
│   └── Raw/
│
├── Notebooks/              # Toàn bộ source code tiền xử lý, phân tích và huấn luyện
│   ├── Phase_1/            # Thu thập và khám phá dữ liệu (EDA)
│   ├── Phase_2/            # Pipeline tiền xử lý và làm sạch dữ liệu
│   └── Phase_3/            # Huấn luyện và Fine-Tuning các mô hình
│       ├── 1.MachineLearning/  # SVM, LightGBM
│       ├── 2.DeepLearning/     # BiLSTM, TextCNN
│       └── 3.FineTuning/       # Code finetune BERT
│
├── Demo/                   # Giao diện Frontend Dashboard
│   ├── index.html          # Bố cục giao diện Neo-brutalism
│   ├── style.css           # CSS tùy chỉnh
│   └── script.js           # Logic tích hợp API và biểu đồ Chart.js
│
└── README.md               # File tài liệu hướng dẫn
```

---

## Hiệu Năng Mô Hình

| Mô Hình | F1-Score | Thời Gian Huấn Luyện | Thời Gian Dự Đoán (ms/mẫu) |
|---------|----------|----------------------|----------------------------|
| **BERT** | 84.74% | 1h 27m 30s | 9.89 |
| **BiLSTM** | 82.99% | 28m 36s | 0.66 |
| **TextCNN**| 81.20% | 20m 11s | 0.61 |
| **LightGBM**| 78.47%| 4m 56s | 0.15 |
| **SVM**    | 77.91% | 1m 01s | 0.03 |

Lưu ý: Thời gian dự đoán được đo lường trên môi trường Kaggle GPU (T4 x2). BERT thể hiện độ chính xác cao nhất nhưng tiêu tốn nhiều tài nguyên tính toán hơn, trong khi SVM cung cấp kết quả baseline với tốc độ cực kỳ nhanh.

---

## Tính Năng Nổi Bật

- **Quy Trình Tiền Xử Lý Toàn Diện**: Sử dụng regex để làm sạch dữ liệu nhưng vẫn giữ nguyên kiểu chữ (casing), xử lý kí tự hình ảnh thông minh (bằng `emoji.demojize`), và chuẩn hóa ngôn ngữ mạng xã hội, giúp các mô hình như BERT giữ lại được các sắc thái ngữ cảnh quan trọng.
- **Tự Động Điều Phối API**: Backend FastAPI tự động gửi văn bản cùng lúc đến cả 5 mô hình đang được nạp sẵn, cho phép so sánh kết quả chạy song song một cách nhanh chóng.
- **Giải Thích AI (XAI)**: Giao diện cung cấp biểu đồ phân phối xác suất, giúp hiểu rõ hơn về mức độ tự tin của từng mô hình.
- **Hệ Thống Cảnh Báo Sớm**: Tự động kích hoạt cảnh báo trên giao diện khi văn bản được phân loại là "Suicidal" với độ tự tin cao.
