# Quy Trình Thi AIC 2026 — Batch 1 (Data BTC)

## Nguyên tắc lưu trữ

```
data/btc/          ← TẤT CẢ data BTC cấp + objects BTC
data/processed/    ← KẾT QUẢ CUỐI: FAISS index + objects nhóm detect
```

## Tổng quan luồng

```
data/btc/
├── clip-features/*.npy    ──┐
├── map-keyframes/*.csv      ├──► python run.py --stage build-btc
├── Keyframes/<video>/       │         │
├── Objects/<video>/*.json   ┘         ▼
└── media-info/*.json             data/processed/
                                  ├── index_btc.faiss       ← FAISS index
                                  └── frame_index_btc.json  ← frame map

data/btc/Keyframes/<video>/*.jpg
         │
         ▼ (Colab — detect objects bổ sung cho video thiếu)
         python -m src.indexing.detect_objects --source btc --model yolo
         │
         ▼
data/processed/objects/<video>/*.json   ← fallback khi BTC chưa có objects

Khi query:
  FAISS search → _load_objects() đọc ưu tiên:
    1. data/btc/Objects/<video>/<n:03d>.json  (BTC format)
    2. data/processed/objects/<video>/<stem>.json (YOLOv8m fallback)
```

---

## BƯỚC 1 — Tạo thư mục và chép data BTC (Local, ~5 phút)

### 1.1 Tạo cấu trúc thư mục

```powershell
mkdir data\btc\clip-features
mkdir data\btc\map-keyframes
mkdir data\btc\Keyframes
mkdir data\btc\Objects
mkdir data\btc\media-info
```

### 1.2 Chép file BTC vào đúng thư mục

| File/Thư mục BTC nhận được | Chép vào |
|---|---|
| `L25_V001.npy`, `L25_V002.npy`... | `data\btc\clip-features\` |
| `L25_V001.csv`, `L25_V002.csv`... | `data\btc\map-keyframes\` |
| Thư mục `L25_V001\`, `L25_V002\`... (ảnh keyframe) | `data\btc\Keyframes\` |
| Thư mục `L25_V004\`... (object JSON BTC detect) | `data\btc\Objects\` |
| `L25_V001_btc.json`... (media info) | `data\btc\media-info\L25_V001.json` (bỏ `_btc`) |

> ⚠️ Nếu BTC đặt tên có `_btc` → đổi tên trước khi chép:

```powershell
# Đổi tên hàng loạt (bỏ "_btc" trong tên file .csv và .json)
Get-ChildItem "data\btc\map-keyframes\" -Filter "*_btc.csv" |
    Rename-Item -NewName { $_.Name -replace '_btc','' }
Get-ChildItem "data\btc\media-info\" -Filter "*_btc.json" |
    Rename-Item -NewName { $_.Name -replace '_btc','' }
```

### 1.3 Cấu trúc sau khi chép xong

```
data/btc/
├── clip-features/
│   ├── L25_V001.npy      ← BTC encode CLIP
│   └── L25_V015.npy
├── map-keyframes/
│   ├── L25_V001.csv      ← frame map (n, pts_time, fps, frame_idx)
│   └── L25_V015.csv
├── Keyframes/
│   ├── L25_V001/
│   │   ├── 001.jpg
│   │   └── ...
│   └── L25_V015/
├── Objects/
│   └── L25_V004/         ← BTC detect sẵn (nếu có)
│       ├── 001.json
│       └── ...
└── media-info/
    └── L25_V001.json

data/processed/            ← TRỐNG lúc đầu, sẽ được tạo tự động
```

---

## BƯỚC 2 — Build FAISS Index (Local, ~2-3 phút)

```powershell
python run.py --stage build-btc
```

**Kết quả trong `data/processed/`:**
```
data/processed/
├── index_btc.faiss           ← FAISS index chính (dùng khi thi)
└── frame_index_btc.json      ← mapping: faiss_idx → video_id, n, pts_time
```

**Kiểm tra:**
```powershell
py -c "import faiss; idx = faiss.read_index('data/processed/index_btc.faiss'); print(f'OK: {idx.ntotal} vectors')"
```

---

## BƯỚC 3 — Detect Objects Bổ Sung (Colab, ~1-2 phút/video)

> Chỉ cần cho video BTC chưa có objects trong `data/btc/Objects/`.
> Kết quả lưu vào `data/processed/objects/` làm fallback.

### 3.1 Chạy trên Colab

```python
# Cell 1: Setup
from google.colab import drive
drive.mount('/content/drive')
%cd /content/AI_Challenge_2026
!pip install ultralytics -q
```

```python
# Cell 2: Copy data btc từ Drive lên Colab
import shutil
from pathlib import Path
shutil.copytree(
    "/content/drive/MyDrive/AIC2026/btc",
    "/content/AI_Challenge_2026/data/btc",
    dirs_exist_ok=True
)
print("✅ Copy xong")
```

```python
# Cell 3: Detect objects trên keyframe BTC
# --source btc → đọc từ data/btc/Keyframes/
# Output → data/processed/objects/ (fallback khi BTC chưa có)
!python -m src.indexing.detect_objects --source btc --model yolo --score-threshold 0.3
```

```python
# Cell 4: Nén và lưu về Drive
import shutil
shutil.make_archive("/content/objects_btc", "zip",
                    "/content/AI_Challenge_2026/data/processed", "objects")
shutil.copy("/content/objects_btc.zip",
            "/content/drive/MyDrive/AIC2026/objects_btc.zip")
print("✅ Lưu objects_btc.zip lên Drive xong!")
```

### 3.2 Giải nén về máy Local

```powershell
Expand-Archive -Path objects_btc.zip -DestinationPath data\processed\ -Force
```

**Kết quả cuối trong `data/processed/`:**
```
data/processed/
├── index_btc.faiss           ← FAISS index
├── frame_index_btc.json      ← frame map
└── objects/
    ├── L25_V001/
    │   ├── 0000.json         ← YOLOv8m format (fallback)
    │   └── ...
    └── L25_V015/
```

---

## BƯỚC 4 — Chạy Server và Test UI

```powershell
# Bật backend
uvicorn src.backend.api:app --reload --port 8000

# Mở UI
start ui\index.html
```

---

## Khi Query — Cách Retriever Đọc Objects

```
Query → FAISS search → top K frames
                            │
                            ▼ với mỗi frame (video_id, n):
              Thử 1: data/btc/Objects/<video_id>/<n:03d>.json   ← BTC (600 class)
              Thử 2: data/processed/objects/<video_id>/<stem>.json ← YOLOv8m (80 class)
                            │
                            ▼
                        rerank → top 20 → UI
```

---

## Tóm Tắt Lệnh

```powershell
# ── LOCAL ──
python run.py --stage build-btc                    # Build FAISS từ data BTC
uvicorn src.backend.api:app --reload --port 8000   # Chạy server

# ── COLAB ──
python -m src.indexing.detect_objects --source btc --model yolo  # Detect objects
```

---

## Thêm Model Sau (OCR / ASR / Face)

Không ảnh hưởng gì đến luồng hiện tại. Mỗi model chỉ thêm 1 nguồn điểm boost trong retriever:

```
data/btc/
├── (hiện tại) clip-features/, map-keyframes/, Keyframes/, Objects/
├── (sau: OCR)  ocr/<video_id>/<n:03d>.json        ← texts trong frame
└── (sau: ASR)  transcripts/<video_id>.json         ← transcript audio

data/processed/
├── index_btc.faiss       ← không thay đổi
└── objects/              ← không thay đổi
```
