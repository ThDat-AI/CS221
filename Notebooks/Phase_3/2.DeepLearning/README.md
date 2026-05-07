# Phase 3 — Deep Learning (BiLSTM / TextCNN)

---

## Bộ embedding pretrained

Hai bộ vector sau đều là **tiếng Anh**, **300 chiều**, phù hợp với pipeline trong `train.py` (`embed_dim=300`) và bài toán phân loại văn bản Reddit / mạng xã hội dạng tiếng Anh.

### GloVe — `glove.840B.300d.txt`

**GloVe** (Global Vectors) học vector từ **ma trận đồng xuất hiện** từ văn bản web rất lớn; từ không có trong từ điển GloVe phải dựa vào `<unk>` hoặc khởi tạo ngẫu nhiên trong mô hình.

- **Dataset Kaggle:** [glove840b300dtxt](https://www.kaggle.com/datasets/takuok/glove840b300dtxt)  

### FastText — English (Wiki + News)

**FastText** (Bojanowski et al., Facebook) học vector dựa trên **n-gram ký tự**. Ưu điểm: từ hiếm hoặc biến thể hình thái có thể được **gần đúng** qua các phần subword, thường **ổn định** khi dữ liệu có nhiễu chính tả / viết tắt. Bộ **English Wiki + News** trên Kaggle tương ứng các vector English được huấn luyện trên Wikipedia và tin.

- **Dataset Kaggle:** [fastText English Word Vectors — Wiki-News](https://www.kaggle.com/datasets/facebook/fasttext-wikinews)

---

## Kết quả thử nghiệm (Drive)

Toàn bộ results được đặt trên Google Drive.

### Cấu trúc một bản (ví dụ `results/GloVeClean-ChangeParam/TextCNN`)

Mỗi nhánh **cấu hình** (`FastTextClean`, `GloVeClean`, …) chứa các thư mục **mô hình** (`TextCNN`, `BiLSTM`, …). Trong mỗi thư mục mô hình chỉ có:

| Số lượng | Nội dung |
|:--------:|----------|
| 1 | `model_final.pt` — checkpoint cuối |
| 2 | `model_history.json` — loss / macro F1 / learning rate theo epoch; `model_metrics.json` — F1 train–val–test, thời gian chạy, v.v. |
| 3 | `model_report_train.txt`, `model_report_val.txt`, `model_report_test.txt` — classification report |
| 1 thư mục | `plots/` — chứa biểu đồ output |

Ví dụ:

```text
GloVeClean-ChangeParam/
  TextCNN/
    textcnn_final.pt
    textcnn_history.json
    textcnn_metrics.json
    textcnn_report_train.txt
    textcnn_report_val.txt
    textcnn_report_test.txt
    plots/
```

Năm bản (**FastTextClean**, **GloVeClean**, **GloVeRaw**, **NoGloVe**, **GloVeClean-ChangeParam**) đều giống trên; chỉ khác cấu hình embedding / dữ liệu / hyperparameter.

**Liên kết Drive:**

[**Google Drive**](https://drive.google.com/drive/folders/1-HiNHTcO0u_w9KpsdM03n5r-kLueSc9A?usp=sharing)

---

## Năm bản thử nghiệm (artifacts)

Mỗi bản tương ứng một cấu hình embedding / dữ liệu / siêu tham số. Tên thư mục trên Drive khớp các nhãn sau để đối chiếu.

### 1. `FastTextClean`

- **Embedding:** FastText (thay cho GloVe).
- **Dữ liệu:** Bản **clean** (pipeline Phase 2).
- **Ý nghĩa:** So sánh chất lượng nhúng FastText so với GloVe trên cùng dữ liệu đã chuẩn hoá.

### 2. `GloVeClean`

- **Embedding:** GloVe pretrained.
- **Dữ liệu:** **Clean**.
- **Ý nghĩa:** Baseline deep learning “chuẩn” so với pipeline làm sạch đầy đủ.

### 3. `GloVeRaw`

- **Embedding:** GloVe pretrained.
- **Dữ liệu:** **Raw** (chưa qua pipeline clean của Phase 2).
- **Ý nghĩa:** Đo lợi ích của bước làm sạch so với GloVeClean.

### 4. `NoGloVe`

- **Embedding:** Không dùng GloVe hay FastText — embedding học từ đầu (random init theo code).
- **Dữ liệu:** Tuỳ bản bạn đã chạy (thường là clean để so sánh công bằng).
- **Ý nghĩa:** Kiểm tra mức độ phụ thuộc vào nhúng pretrained.

### 5. `GloVeClean-ChangeParam`

- **Embedding:** GloVe.
- **Dữ liệu:** **Clean**.
- **Khác biệt:** Siêu tham số không dùng mặc định của script; cụ thể:

```json
{
  "embed_dim": 300,
  "hidden_dim": 192,
  "max_len": 300,
  "min_freq": 2,
  "batch_size": 64,
  "epochs": 40
}
```

So với baseline mặc định trong script:

```json
{
  "hidden_dim": 128,
  "max_len": 256,
  "min_freq": 2,
  "batch_size": 64,
  "epochs": 40
}
```

**Tóm tắt thay đổi:** `hidden_dim` 128 → 192, `max_len` 256 → 300; các mục khác (`embed_dim`, `min_freq`, `batch_size`, `epochs`) giữ nguyên — mục tiêu là kiểm tra ảnh hưởng của **chiều ẩn LSTM lớn hơn** và **độ dài câu tối đa dài hơn**.

---

## Mã nguồn trong repo

- `train.py` — huấn luyện BiLSTM / TextCNN từ CLI.
- `data_utils.py` — vocab, tokenize, placeholder, xử lý emoji dạng `demojize` cho DL.
- `models.py` — kiến trúc BiLSTM + attention, TextCNN.
- `Train_*.ipynb` — notebook chạy trên Kaggle / môi trường tương tự.

Chi tiết chạy local hoặc trên GPU xem thêm comment trong `train.py` và notebook.
