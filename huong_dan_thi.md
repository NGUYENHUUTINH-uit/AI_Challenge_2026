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

> **Khi nào cần:** Khi BTC chưa cung cấp objects cho một số video trong `data/btc/Objects/`.
> Kết quả detect lưu vào `data/processed/objects/` làm **fallback** — retriever tự ưu tiên
> BTC objects trước, chỉ dùng fallback khi không có.

> **Tình trạng:** Nhiều video thiếu objects (rải rác L27-L30), tổng ~30k keyframes.
> Vì dữ liệu lớn (~6 GB anh), **nen va detect tung lesson** de tranh timeout Colab.

### 3.0 Kiểm tra video nào đang thiếu objects (Local)

```powershell
python -c "
from pathlib import Path
btc_kf  = Path('data/btc/Keyframes')
btc_obj = Path('data/btc/Objects')
kf  = set(v.name for v in btc_kf.iterdir()  if v.is_dir())
obj = set(v.name for v in btc_obj.iterdir() if v.is_dir())
missing = sorted(kf - obj)
print(f'Thieu objects: {len(missing)} videos')
for v in missing: print(' ', v)
"
```

### 3.1 Nén file cần thiết (Local — PowerShell)

> Nén **2 loại file** rồi upload lên Google Drive.

**Lần đầu tiên — nén source code (chỉ làm 1 lần):**
```powershell
# Dung Python de nen (tranh loi duong dan Windows tren Linux/Colab)
python -c "import shutil; shutil.make_archive('src_code', 'zip', '.', 'src')"
```

**Mỗi lesson cần detect — nén keyframes:**
```powershell
# Thay LXX bang lesson can detect, vi du: L27, L28, L29, L30
python -c "
import zipfile
from pathlib import Path
lesson = 'LXX'
kf_dir = Path('data/btc/Keyframes')
with zipfile.ZipFile(f'keyframes_{lesson}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for vd in kf_dir.iterdir():
        if vd.is_dir() and vd.name.startswith(lesson):
            for img in vd.iterdir():
                zf.write(img, f'{vd.name}/{img.name}')
print('Done! keyframes_' + lesson + '.zip')
"
```

> **Upload lên Google Drive** vào đúng thư mục sau (tạo nếu chưa có):
> ```
> My Drive/
> └── AIC2026/
>     ├── src_code.zip          ← upload 1 lan duy nhat
>     ├── keyframes_L27.zip     ← upload tung lesson
>     ├── keyframes_L28.zip
>     ├── keyframes_L29.zip
>     └── keyframes_L30.zip
> ```
> Cách upload: [drive.google.com](https://drive.google.com) → thư mục `AIC2026` → kéo thả file vào.

### 3.2 Chạy trên Colab

> **Cell 1 chạy 1 lần.** Sau đó mỗi lesson: đổi `LESSON` ở Cell 2 → chạy Cell 2→3→4→5 → lặp.

```python
# Cell 1: Setup — chay 1 lan duy nhat
from google.colab import drive
drive.mount('/content/drive')

import zipfile, os

# Giai nen source code tu Drive (khong can git clone)
os.makedirs('/content/AIC2026', exist_ok=True)
with zipfile.ZipFile('/content/drive/MyDrive/AIC2026/src_code.zip') as z:
    z.extractall('/content/AIC2026')

%cd /content/AIC2026

os.makedirs('data/btc/Keyframes', exist_ok=True)
os.makedirs('data/processed/objects', exist_ok=True)
!pip install ultralytics tqdm pillow -q
print("OK Setup xong!")
```

```python
# Cell 2: Giai nen keyframes — DOI LESSON MOI LAN
LESSON = "L30"   # L27 / L28 / L29 / L30 / ...

import zipfile, os
with zipfile.ZipFile(f"/content/drive/MyDrive/AIC2026/keyframes_{LESSON}.zip") as z:
    z.extractall("data/btc/Keyframes")
videos = [v for v in os.listdir("data/btc/Keyframes") if v.startswith(LESSON)]
print(f"OK Giai nen xong: {len(videos)} videos")
```

```python
# Cell 3: Detect objects
# Chay file truc tiep (tranh loi module not found)
import subprocess
result = subprocess.run(
    ['python', 'src/indexing/detect_objects.py',
     '--source', 'btc', '--model', 'yolo', '--score-threshold', '0.3'],
    cwd='/content/AIC2026'
)
print("Done! Exit code:", result.returncode)
```

```python
# Cell 4: Kiem tra nhanh
from pathlib import Path
obj_dir = Path("data/processed/objects")
done  = [v.name for v in obj_dir.iterdir() if v.is_dir()]
total = sum(len(list(v.iterdir())) for v in obj_dir.iterdir() if v.is_dir())
print(f"OK {len(done)} videos | {total} file JSON")
```

```python
# Cell 5: Nen ket qua -> luu Drive -> don dep de chay lesson tiep
import shutil, os

shutil.make_archive(f"/content/objects_{LESSON}", "zip", "data/processed", "objects")
shutil.copy(f"/content/objects_{LESSON}.zip",
            f"/content/drive/MyDrive/AIC2026/objects_{LESSON}.zip")
print(f"OK Luu objects_{LESSON}.zip len Drive xong!")

shutil.rmtree("data/btc/Keyframes", ignore_errors=True)
shutil.rmtree("data/processed/objects", ignore_errors=True)
os.makedirs("data/btc/Keyframes", exist_ok=True)
os.makedirs("data/processed/objects", exist_ok=True)
print("-> Doi LESSON o Cell 2 va chay lai tu Cell 2")
```

### 3.3 Giải nén kết quả về Local (sau mỗi lesson)

```powershell
# Thay <LXX> bang lesson vua detect xong
Expand-Archive -Path "objects_<LXX>.zip" -DestinationPath "data\processed\" -Force
```

**Kết quả cuối trong `data/processed/`:**
```
data/processed/
├── index_btc.faiss           <- FAISS index
├── frame_index_btc.json      <- frame map
└── objects/
    ├── <LXX>_V001/
    │   ├── 0000.json         <- YOLOv8m format (fallback)
    │   └── ...
    └── <LXX>_Vxxx/
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
