
# AI Challenge 2026 — Hướng Dẫn Đầy Đủ

## 1. Hướng Dẫn Cài Đặt Môi Trường

### Yêu cầu hệ thống
- Python **3.10 – 3.12** đã được cài đặt (kiểm tra: `python --version`)
- Git đã được cài đặt (kiểm tra: `git --version`)
- Windows 10/11

### Bước 0 — Clone repo về máy (làm 1 lần đầu tiên)

```powershell
git clone https://github.com/NGUYENHUUTINH-uit/AI_Challenge_2026.git
cd AI_Challenge_2026
```

> ⚠️ Nếu repo là **private**, bạn cần có quyền truy cập hoặc dùng token (xem Mục 8).

---

### Bước 1 — Lưu video

> Bỏ video `.mp4` được giao vào thư mục `data\raw_videos\` trước khi chạy ingest.

---

### Bước 2 — Cài đặt môi trường (chạy 1 lần)

#### Cách A — Tự động 

```powershell
setup.bat
```

Script sẽ tự động:
1. Tạo môi trường ảo `venv/`
2. Kích hoạt `venv`
3. Cài CLIP từ GitHub
4. Cài toàn bộ thư viện trong `requirements.txt` (bao gồm `opencv-python`, `torch`, `faiss-cpu`, v.v.)

#### Cách B — Thủ công (nếu `setup.bat` bị lỗi)

```powershell
# 1. Tạo và kích hoạt venv
python -m venv venv
venv\Scripts\activate.bat

# 2. Nâng cấp pip trước
pip install --upgrade pip

# 3. Cài CLIP (bắt buộc cài trước requirements.txt)
pip install git+https://github.com/openai/CLIP.git

# 4. Cài toàn bộ thư viện còn lại
pip install -r requirements.txt
```

---

### Bước 3 — Kiểm tra cài đặt thành công

> ⚠️ **QUAN TRỌNG**: Phải **kích hoạt venv trước** rồi mới chạy lệnh kiểm tra!

```powershell
# Kích hoạt venv (phải làm mỗi phiên làm việc mới)
venv\Scripts\activate.bat

# Kiểm tra opencv (cv2) — hay bị lỗi nhất
python -c "import cv2; print('cv2 OK, version:', cv2.__version__)"

# Kiểm tra torch
python -c "import torch; print('torch OK, version:', torch.__version__)"

# Kiểm tra CLIP
python -c "import clip; print('CLIP OK')"

# Kiểm tra faiss
python -c "import faiss; print('faiss OK')"
```

Nếu tất cả in ra `OK` → môi trường đã sẵn sàng.

---

### ❌ Xử lý lỗi thường gặp

#### Lỗi: `ModuleNotFoundError: No module named 'cv2'`

**Nguyên nhân**: Bạn đang chạy Python **ngoài venv** (chưa kích hoạt môi trường ảo), hoặc `setup.bat` bị lỗi giữa chừng nên `opencv-python` chưa được cài.

**Cách sửa**:

```powershell
# Bước 1: Kích hoạt venv TRƯỚC
venv\Scripts\activate.bat

# Bước 2: Kiểm tra xem pip đang trỏ vào đâu
where pip
# Phải thấy đường dẫn có chứa "venv", ví dụ:
# D:\AIC_2026\AI_Challenge_2026\venv\Scripts\pip.exe

# Bước 3: Cài lại opencv
pip install opencv-python

# Bước 4: Kiểm tra lại
python -c "import cv2; print('cv2 OK:', cv2.__version__)"
```

#### Lỗi: `ModuleNotFoundError: No module named 'clip'`

```powershell
venv\Scripts\activate.bat
pip install git+https://github.com/openai/CLIP.git
```

#### Lỗi: `torch` không nhận GPU

Cài lại torch theo đúng CUDA của máy tại [https://pytorch.org](https://pytorch.org):

```powershell
# Ví dụ CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### Bước 4 — Mỗi lần làm việc, kích hoạt lại venv

```powershell
venv\Scripts\activate.bat
```

> Dấu hiệu kích hoạt thành công: dòng lệnh bắt đầu bằng `(venv)`.

---

### Lưu ý khác

> ⚠️ **torch / CUDA**: Nếu máy có GPU, cài `torch` theo đúng phiên bản CUDA tại [https://pytorch.org](https://pytorch.org) **trước** khi chạy `setup.bat`.

> ⚠️ **API Keys**: Tạo file `.env` ở thư mục gốc và điền các key cần thiết (không commit file này lên Git):
> ```
> OPENAI_API_KEY=sk-...
> ANTHROPIC_API_KEY=...
> GOOGLE_API_KEY=...
> ```

---

## 7. Pipeline Xử Lý Data — Kịch Bản Nhiều Máy (Bản đầy đủ)

> BTC gửi nhiều video → chia đều cho từng máy → mỗi máy xử lý phần của mình → gộp lại build FAISS


---

### Tổng quan luồng (đã bổ sung)

```
BTC gửi N video (.mp4)
    │
    ├──[Người A] 50 video → Máy A
    ├──[Người B] 50 video → Máy B  ← Mỗi máy chạy các bước giống nhau (xem bên dưới)
    ├──[Người C] 50 video → Máy C
    └──[Người D] 50 video → Máy D
                                   │
                    ┌──────────────┘
                    ▼
        [Người A] Gộp .npy từ tất cả máy
            └── python run.py --stage build-self
                    → index_self.faiss   ← SẴN SÀNG THI
                    │
                    ▼
            python run.py --stage verify
                    → Đối chiếu với data BTC cho sẵn (nếu có)
                    → Xác nhận pipeline đáng tin trước khi dùng khi thi thật
```

---

### Bước 1 — Chia video (Người A làm 1 lần)


---

### Bước 2 — Mỗi máy xử lý phần của mình (LOCAL)

> Tất cả máy chạy **cùng các lệnh sau**, theo đúng thứ tự — không được đảo lộn.

```powershell
# 0. Kích hoạt môi trường ảo (PHẢI làm đầu tiên mỗi phiên làm việc)
venv\Scripts\activate

# Kiểm tra (venv) xuất hiện ở đầu dòng lệnh trước khi tiếp tục!

# 1. Bỏ video được giao vào đây:
#    data\raw_videos\

# 2. Trích keyframe (~5-15 phút/video tùy độ dài)
python -m src.indexing.ingest --source raw

# 3. Nén keyframes để upload lên Colab
python -c "import shutil; shutil.make_archive('keyframes', 'zip', 'data/processed', 'keyframes')"
```

Sau đó **upload `keyframes.zip` lên Google Drive** → sang Bước 3 (Colab) để chạy Encode CLIP + Object Detection. (nhớ tạo thư mục tên **AIC2026** trên GG Drive (của bạn) để đưa file nén vào)

---

### Bước 3 — Encode CLIP trên Google Colab (GPU) — Xem chi tiết tại [Section 8](#8-google-colab---5-cell-encode-clip) bên dưới.

---

### Bước 4 — Sau khi download kết quả từ Colab về máy

Không đổi:
Giải nén file `.zip` (chứa cả `clip-features` và `objects` từ Colab trả về):
```bash
python -c "import shutil; shutil.unpack_archive('colab_results.zip', 'data/processed')"
```

---

### 📩 Cách nộp kết quả cho Trưởng Nhóm

Sau khi hoàn tất Bước 4, mỗi thành viên **gom đúng 3 mục sau** và gửi cho Trưởng Nhóm (qua Drive/Zalo/USB):
1. `colab_results.zip` (Tải từ Drive về).
2. Thư mục `data/processed/map-keyframes/` (Nằm trên máy tính).
3. Thư mục `data/processed/media-info/` (Nằm trên máy tính).

*(Lưu ý: Bạn có thể copy 3 mục này ra một thư mục riêng, nén lại thành `KetQua_TenBan.zip` rồi gửi cho gọn).*

---

### Bước 5 — Trưởng nhóm (Người A) gộp tất cả, build FAISS

```powershell
# Trưởng nhóm giải nén file của tất cả đồng đội,
# Copy đè (merge) tất cả vào đúng thư mục `data/processed/` trên máy tính của mình.

# Sau khi "hội tụ" đủ data của toàn đội, Trưởng nhóm chạy lệnh:
python run.py --stage build-self

# Kết quả:
#   data\processed\index_self.faiss        ← FAISS sẵn sàng thi
#   data\processed\frame_index_self.json   ← mapping global → {video_id, frame_idx (từ map-keyframes.csv)}
#   data\processed\clip_features_self.npy  ← features matrix tổng
```

**Kiểm tra bắt buộc:** mở `frame_index_self.json`, xác nhận field lưu đúng tên **`frame_idx`** (số frame thật lấy từ `map-keyframes/*.csv`), không phải số tự đếm theo thứ tự file `.npy`. Đây là con số BTC dùng để chấm điểm — sai field này thì dù tìm đúng khung hình vẫn bị chấm 0.

---

### 🆕 Bước 6 — Kiểm chứng pipeline trước khi tin dùng

```powershell
python run.py --stage verify
```

So sánh vector tự encode (từ `run.py --stage encode`) với vector BTC cho sẵn (`clip-features-32/*.npy`) trên cùng ảnh — in ra cosine similarity trung bình.

- **> 0.95**: pipeline đúng chuẩn, yên tâm dùng khi thi thật (lúc đó BTC chỉ đưa video, không có gì để đối chiếu nữa).
- **< 0.85**: dừng lại, kiểm tra lại bước `filter`/`ingest` — khả năng cao đang bật letterbox/normalize sai cách trước khi encode.

**Chỉ chạy được nếu có data BTC batch tham khảo để đối chiếu** — không áp dụng được khi xử lý video thi thật (vì lúc đó không có "đáp án mẫu").

---

### Cấu trúc thư mục sau khi hoàn tất (đã cập nhật)

```
data\processed\
├── keyframes\<video_id>\0000.jpg ...        ← keyframe thô (trước lọc)
├── keyframes_clean\<video_id>\...           ← 🆕 sau filter + group-scenes (encode từ đây)
├── metadata_clean.json                       ← 🆕 có blur_score, is_clean, scene_id, is_representative
├── map-keyframes\<video_id>.csv              ← frame map (frame_idx thật)
├── media-info\<video_id>.json                ← fps, duration (auto)
├── objects\<video_id>\0000.json ...           ← object detection (BTC cho hoặc tự detect)
├── clip-features\<video_id>.npy               ← CLIP features (từ Colab)
│
├── index_self.faiss                          ← FAISS tổng ← SẴN SÀNG THI
└── frame_index_self.json                     ← mapping (dùng đúng frame_idx)
```

---

## 8. Google Colab — 5 Cell Encode CLIP

> Chạy sau Bước 2 (đã có `keyframes.zip` — **chứa `keyframes_clean/`, không phải `keyframes/` thô**).
> Yêu cầu: **Runtime → Change runtime type → T4 GPU**

---

### Cell 1 — Clone code từ GitHub (không đổi)

> Yêu cầu: **Runtime → Change runtime type → T4 GPU**

---



> **Chuẩn bị token 1 lần duy nhất** (làm trước khi chạy Colab):
>
> 1. Vào GitHub → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
> 2. Bấm **Generate new token (classic)**
> 3. Đặt tên `aic2026-colab`, thời hạn **30 ngày**
> 4. Tick đúng **1 ô**: `repo` (Full control of private repositories)
> 5. Bấm **Generate token** → **copy ngay** (chỉ hiện 1 lần!)
> 6. Trong Colab: bấm icon 🔑 **Secrets** (thanh bên trái) → **Add new secret**
>    - Name: `GITHUB_TOKEN`
>    - Value: dán token vừa copy
>    - Bật **Notebook access** → Save

### Cell 1 — Clone code từ GitHub

```python
# ===== CELL 1: Lấy token từ Colab Secrets + clone repo =====
from google.colab import userdata

GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')   # lấy từ Colab Secrets (an toàn, không lộ token)
print(f"Token OK: {len(GITHUB_TOKEN)} ký tự")  # chỉ in độ dài, không in nội dung

REPO   = "NGUYENHUUTINH-uit/AI_Challenge_2026"
BRANCH = "main"   # ← thay tên nhánh của bạn nếu khác

%cd /content
!rm -rf AI_Challenge_2026
!git clone https://{GITHUB_TOKEN}@github.com/{REPO}.git
%cd AI_Challenge_2026
!git checkout {BRANCH}
!ls src/indexing   # kiểm tra code đã clone đúng
```

---

### Cell 2 — Cài thư viện

```python
!pip install -q ftfy regex tqdm Pillow numpy pandas
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q faiss-cpu          # cần thiết cho indexer.py
!pip install -q ultralytics        # YOLOv8m — nhanh hơn Faster R-CNN ~8x

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ KHÔNG CÓ GPU'}")
```

---

### Cell 3 — Mount Drive và giải nén keyframes ( sửa tên thư mục)

```python
from google.colab import drive
drive.mount('/content/drive')

ZIP_PATH = "/content/drive/MyDrive/AIC2026/keyframes.zip"

!unzip -q -o "{ZIP_PATH}" -d /content/AI_Challenge_2026/data/processed/

import os
kf_dir    = "/content/AI_Challenge_2026/data/processed/keyframes"
n_videos  = len([d for d in os.listdir(kf_dir) if os.path.isdir(f"{kf_dir}/{d}")])
n_frames  = sum(len(os.listdir(f"{kf_dir}/{d}")) for d in os.listdir(kf_dir)
                if os.path.isdir(f"{kf_dir}/{d}"))
print(f"✅ {n_videos} video | {n_frames} frames")
```

---

### Cell 4 — Encode CLIP và Object Detection (Tối ưu cho T4 GPU)

```python
import time
start = time.time()

# ===== 🚀 THÔNG SỐ TỐI ƯU CHO T4 GPU 16GB =====
# batch-size 128: tăng throughput CLIP ~4x so với 32
# (fp16 autocast đã được bật tự động trong code)

# 1. Encode CLIP vector
print("Đang chạy Encode CLIP...")
!python run.py --stage encode --batch-size 128

# 2. Object detection (YOLOv8m — nhanh hơn Faster R-CNN ~8x, mAP cao hơn)
print("\nĐang chạy Object Detection...")
!python run.py --stage detect --obj-model yolo

elapsed = time.time() - start
print(f"\n✅ Tổng thời gian: {elapsed/60:.1f} phút")
````


### Cell 5 — Nén kết quả và lưu về Drive

```python
import shutil
import os

# Gom cả clip-features và objects vào một thư mục chung để nén
os.makedirs("/content/results", exist_ok=True)
!cp -r /content/AI_Challenge_2026/data/processed/clip-features /content/results/
!cp -r /content/AI_Challenge_2026/data/processed/objects /content/results/

# Nén thư mục kết quả
shutil.make_archive(
    "/content/colab_results",   # tên file zip
    "zip",
    "/content/results"
)

# Lưu về Drive
shutil.copy(
    "/content/colab_results.zip",
    "/content/drive/MyDrive/AIC2026/colab_results.zip"
)
print("✅ Đã lưu colab_results.zip lên Drive!")
print("   → Tải về máy Local rồi giải nén vào data/processed/, sau đó chạy: python run.py --stage build-self")
```