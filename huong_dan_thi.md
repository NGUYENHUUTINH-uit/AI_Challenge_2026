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

> Nén keyframes rồi upload lên Google Drive.

**Mỗi lesson cần detect — nén keyframes (dùng Python để đảm bảo forward slash):**
```powershell
# QUAN TRONG: chay lenh nay tu thu muc goc project (noi co thu muc src\ va data\)
cd d:\AIC_2026\AI_Challenge_2026

# Thay L27 bang lesson can detect
python -c "
import zipfile
from pathlib import Path
lesson = 'L26'
kf_dir = Path('data/btc/Keyframes')
with zipfile.ZipFile(f'keyframes_{lesson}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for vd in kf_dir.iterdir():
        if vd.is_dir() and vd.name.startswith(lesson):
            for img in vd.iterdir():
                # Dung forward slash tuong minh, tranh loi backslash tren Linux/Colab
                zf.write(img, vd.name + '/' + img.name)
print('Done! keyframes_' + lesson + '.zip')
"
```

> **Upload lên Google Drive** vào đúng thư mục sau (tạo nếu chưa có):
> ```
> My Drive/
> └── AIC2026/
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
# Lay token tu Colab Secrets (an toan, khong lo token)
from google.colab import userdata, drive

GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
print(f"Token OK: {len(GITHUB_TOKEN)} ky tu")

REPO   = "NGUYENHUUTINH-uit/AI_Challenge_2026"
BRANCH = "feature-A7"   # <- nhánh đang chứa code tối ưu mới nhất

# Clone code tu GitHub
%cd /content
!rm -rf AIC2026
!git clone https://{GITHUB_TOKEN}@github.com/{REPO}.git AIC2026
%cd AIC2026
!git checkout {BRANCH}
!ls src/indexing   # kiem tra code da clone dung

# Mount Drive de truy cap keyframes
drive.mount('/content/drive')

import os
os.makedirs('data/btc/Keyframes', exist_ok=True)
os.makedirs('data/processed/objects', exist_ok=True)
!pip install ultralytics tqdm pillow -q
print("OK Setup xong!")
```

```python
# Cell 2: Giai nen keyframes — DOI LESSON MOI LAN
LESSON = "L27"   # L27 / L28 / L29 / L30 / ...

import zipfile, os
from pathlib import Path

kf_base = Path("data/btc/Keyframes")

with zipfile.ZipFile(f"/content/drive/MyDrive/AIC2026/keyframes_{LESSON}.zip") as z:
    for member in z.namelist():
        # Chuan hoa backslash (Windows ZIP) -> forward slash (Linux)
        norm = member.replace('\\', '/')
        out_path = kf_base / norm
        if norm.endswith('/'):
            out_path.mkdir(parents=True, exist_ok=True)
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(z.read(member))

videos = [v for v in os.listdir(kf_base) if (kf_base / v).is_dir() and v.startswith(LESSON)]
print(f"OK Giai nen xong: {len(videos)} videos")
```

```python
# Cell 3: Detect objects (Nạp model 1 lần, hiển thị tiến độ % trực tiếp)
%cd /content/AIC2026
!python src/indexing/detect_objects.py --source btc --model yolo --score-threshold 0.3
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
Expand-Archive -Path "objects_LXX.zip" -DestinationPath "data\processed\" -Force
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

## BƯỚC 5 — Chạy OCR Phân Tán (nhóm chia nhau chạy)

> OCR giúp hệ thống tìm frame có **số, chữ, thương hiệu** (biển số, logo, năm...).  
> Script tự **resume** nếu bị ngắt — chạy lại lệnh cũ là tiếp tục từ chỗ dừng.

### Yêu cầu dữ liệu

OCR chỉ cần **ảnh keyframe JPG** — **không cần video gốc**.

```
data/btc/Keyframes/   ← 28.64 GB JPG, đã có sẵn từ BTC
├── L21_V001/001.jpg
├── L21_V001/002.jpg
└── ...
```

---

### Cách A — Chạy local (có GPU)

```powershell
# Bước 0: cài EasyOCR (1 lần)
pip install easyocr

# Bước 1: xem kế hoạch chia trước (ví dụ nhóm 3 người)
python -m src.indexing.run_ocr_worker --plan --total 3

# Bước 2: mỗi người chạy phần của mình
python -m src.indexing.run_ocr_worker --part 1 --total 3   # Người 1
python -m src.indexing.run_ocr_worker --part 2 --total 3   # Người 2
python -m src.indexing.run_ocr_worker --part 3 --total 3   # Người 3

# Nếu không có GPU:
python -m src.indexing.run_ocr_worker --part 1 --total 3 --cpu
```

Output mỗi người: `data/processed/ocr_part_X_of_3.json`

```powershell
# Bước 3: copy tất cả file partial vào data/processed/ rồi merge
python -m src.indexing.run_ocr_worker --merge
# → Tạo ra: data/processed/ocr_index.json
```

---

---

### Cách B — Chạy trên Google Colab (GPU T4 miễn phí)

> File notebook: `colab_ocr_worker.ipynb` — upload lên Colab để dùng.

#### B.1 Cấu trúc Google Drive của nhóm

Các file zip đã được upload sẵn tại `MyDrive/AIC2026/`:

```
MyDrive/
└── AIC2026/                    ← thư mục gốc trên Drive
    ├── keyframes_L21.zip       (1.34 GB)
    ├── keyframes_L22.zip       (1.59 GB)
    ├── keyframes_L23.zip       (481.6 MB)
    ├── keyframes_L24.zip       (1.61 GB)
    ├── keyframes_L25.zip       ← các lesson còn lại tương tự
    └── ...
```

> **Không cần làm gì thêm với Drive** — chỉ cần Colab mount vào và unzip.

---

#### B.2 Pull code về Colab (làm đầu mỗi session)

```bash
# Bước 1: Clone nhánh feature-A7 về Colab (lần đầu)
git clone -b feature-A7 https://github.com/NGUYENHUUTINH-uit/AI_Challenge_2026.git /content/AIC_2026

# Bước 2: Nếu đã clone từ session trước → chỉ cần pull
cd /content/AIC_2026
git pull origin feature-A7

# Bước 3: Vào thư mục project
cd /content/AIC_2026
```

> **Kiểm tra nhánh đang đứng:**  
> `git branch` → phải thấy `* feature-A7`

---

#### B.3 Unzip Keyframes từ Drive ra Colab

> Chỉ cần unzip 1 lần cho session đầu tiên. Nếu session bị ngắt và chạy lại → **bỏ qua bước này** nếu `/content/keyframes/` vẫn còn.

```python
# Chạy trong Colab cell:
import zipfile, os
from pathlib import Path

zip_dir = Path('/content/drive/MyDrive/AIC2026')   # thư mục chứa file zip trên Drive
out_dir = Path('/content/keyframes')                # unzip ra đây (Colab local disk)

# Kiểm tra nếu đã unzip từ session trước
if out_dir.exists() and len(list(out_dir.iterdir())) > 0:
    print(f'[Skip] /content/keyframes/ đã có {len(list(out_dir.iterdir()))} thư mục — bỏ qua unzip')
else:
    out_dir.mkdir(exist_ok=True)
    # File tên dạng: keyframes_L21.zip, keyframes_L22.zip, ...
    zip_files = sorted(zip_dir.glob('keyframes_L*.zip'))
    print(f'Tìm thấy {len(zip_files)} file zip: {[f.name for f in zip_files]}')

    for zf in zip_files:
        print(f'Đang unzip {zf.name} ({zf.stat().st_size/1024/1024/1024:.2f} GB) ...')
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(out_dir)
        print(f'  ✅ Xong: {zf.name}')

# Kiểm tra kết quả
n_imgs = len(list(out_dir.rglob('*.jpg')))
print(f'\nTổng ảnh: {n_imgs:,}  (kỳ vọng ~177,000)')
```

Sau khi unzip, cấu trúc sẽ là:
```
/content/keyframes/
├── L21_V001/
│   ├── 001.jpg
│   └── ...
├── L21_V002/
└── ...
```

Sau đó **trỏ project vào thư mục vừa unzip**:
```python
import os
from pathlib import Path

# Tạo symlink: data/btc/Keyframes → /content/keyframes
project_kf = Path('/content/AIC_2026/data/btc/Keyframes')
project_kf.parent.mkdir(parents=True, exist_ok=True)

if project_kf.is_symlink():
    project_kf.unlink()
if not project_kf.exists():
    os.symlink('/content/keyframes', str(project_kf))

print(f'✅ OK: {len(list(project_kf.iterdir()))} video dirs')
```




#### B.4 Chạy OCR

```bash
# Cài EasyOCR (1 lần mỗi session)
pip install easyocr -q

# Xem kế hoạch chia (tuỳ chọn)
cd /content/AIC_2026
python -m src.indexing.run_ocr_worker --plan --total 3

# Chạy phần của mình (GPU T4, ~5-10h)
python -m src.indexing.run_ocr_worker --part 1 --total 3
# Script tự resume nếu bị ngắt — chạy lại lệnh trên là tiếp tục
```

---

#### B.5 Lưu kết quả về Drive (quan trọng — chạy sau khi xong)

```python
import shutil
from pathlib import Path

PART, TOTAL = 1, 3   # chỉnh theo phần của bạn
src = Path(f'/content/AIC_2026/data/processed/ocr_part_{PART}_of_{TOTAL}.json')
dst = Path('/content/drive/MyDrive/AIC2026_data/ocr_results')
dst.mkdir(exist_ok=True)

shutil.copy2(src, dst / src.name)
print(f'Saved: {dst / src.name}  ({src.stat().st_size/1024/1024:.1f} MB)')
```

---

#### B.6 Merge (người được phân công)

Copy tất cả `ocr_part_*.json` từ Drive về `data/processed/`, rồi:

```bash
# Local (Windows)
python -m src.indexing.run_ocr_worker --merge

# Hoặc trên Colab
python -m src.indexing.run_ocr_worker --merge
```

→ Tạo ra `data/processed/ocr_index.json`, **commit lên Git** để cả nhóm dùng được.

---

### Ước tính thời gian (872 video / 177K ảnh chia 3 người)

| Người | Video | Ảnh | GPU Colab T4 | CPU local |
|-------|-------|-----|-------------|-----------|
| 1 | 1–291 | ~89K | ~10h | ~37h |
| 2 | 292–582 | ~46K | ~5h | ~19h |
| 3 | 583–872 | ~50K | ~6h | ~21h |

> **Mẹo Colab**: Nếu sợ session bị ngắt sau 12h, đặt `--total 6` chia nhỏ hơn.  
> Script sẽ resume đúng chỗ khi chạy lại.

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
├── ocr_index.json        ← output của bước OCR phân tán
└── objects/              ← không thay đổi
```
