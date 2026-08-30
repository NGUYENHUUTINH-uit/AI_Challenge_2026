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

## BƯỚC 3.5 — Build CLIP ViT-L/14 Index (Colab, ~30 phút T4)

> **Mục đích:** Nâng cấp ViT-B/32 (512d) → ViT-L/14 (768d) → +30% accuracy.
> **Retriever tự dùng nếu file tồn tại**, không cần sửa code.

### 3.5.1 Chạy trên Colab

**Cell 1 — Setup (chạy 1 lần):**
```python
from google.colab import userdata, drive
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
REPO   = "NGUYENHUUTINH-uit/AI_Challenge_2026"
BRANCH = "feature-A7"
%cd /content
!rm -rf AIC2026
!git clone https://{GITHUB_TOKEN}@github.com/{REPO}.git AIC2026
%cd AIC2026
!git checkout {BRANCH}
drive.mount('/content/drive')
import os
os.makedirs('data/processed', exist_ok=True)
os.makedirs('data/btc/Keyframes', exist_ok=True)
!pip install openai-clip faiss-gpu Pillow tqdm -q
print("Setup OK!")
```

**Cell 2 — Copy frame map từ Drive:**
```python
import shutil
shutil.copy("/content/drive/MyDrive/AIC2026/frame_index_btc.json",
            "data/processed/frame_index_btc.json")
print("frame_index_btc.json OK!")
```

**Cell 3 — Giải nén keyframes (dùng lại zip từ bước detect objects):**
```python
import zipfile
from pathlib import Path
kf_base = Path("data/btc/Keyframes")
for lesson in ["L21","L22","L23","L24","L25","L26","L27","L28","L29","L30"]:
    zip_path = f"/content/drive/MyDrive/AIC2026/keyframes_{lesson}.zip"
    if not Path(zip_path).exists():
        print(f"Skip {lesson} (no zip)")
        continue
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            norm = member.replace('\\', '/')
            out_path = kf_base / norm
            if norm.endswith('/'): out_path.mkdir(parents=True, exist_ok=True)
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(z.read(member))
    print(f"OK {lesson}")
```

**Cell 4 — Build index (~30 phút T4, tự lưu checkpoint mỗi 5000 frame):**
```python
!python src/indexing/build_vitl14_index.py \
    --model ViT-L/14 \
    --batch-size 256 \
    --drive-output /content/drive/MyDrive/AIC2026/ \
    --output data/processed/index_btc_vitl14.faiss
# Nếu Colab crash giữa chừng: thêm --resume vào lệnh trên
```

**Cell 5 — Kiểm tra:**
```python
import faiss
idx = faiss.read_index("data/processed/index_btc_vitl14.faiss")
print(f"OK: {idx.ntotal} vectors, dim={idx.d}")  # 177321 vectors, dim=768
```

### 3.5.2 Kích hoạt Local

```powershell
# Copy file vừa tải về từ Drive vào đúng thư mục
copy "Downloads\index_btc_vitl14.faiss" "data\processed\index_btc_vitl14.faiss"

# Restart server — log sẽ hiện "ViT-L/14" khi load thành công
uvicorn src.backend.api:app --reload --port 8000
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

#### B.2 Mount Drive & Kéo code về Colab (Cell 1 & Cell 2)

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2: Kéo code từ nhánh feature-A7 (Đọc Token từ mục Secrets 🔑)
import os
from google.colab import userdata

try:
    # Tự động lấy token từ mục Secrets (icon chìa khóa 🔑 bên trái Colab: Name = GITHUB_TOKEN)
    GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
    REPO_URL = f"https://{GITHUB_TOKEN}@github.com/NGUYENHUUTINH-uit/AI_Challenge_2026.git"
except Exception:
    # Nếu repo là PUBLIC hoặc không cài Secrets:
    REPO_URL = "https://github.com/NGUYENHUUTINH-uit/AI_Challenge_2026.git"

if not os.path.exists('/content/AIC_2026'):
    !git clone -b feature-A7 {REPO_URL} /content/AIC_2026
else:
    !cd /content/AIC_2026 && git pull origin feature-A7

os.chdir('/content/AIC_2026')
print('✅ Code sẵn sàng tại /content/AIC_2026')
```

---

#### B.3 Unzip Keyframes từ Drive ra Colab (Cell 3)

> Chỉ cần unzip 1 lần. Nếu session trước đã unzip thì đoạn code này sẽ tự động bỏ qua (Skip).

```python
# Cell 3: Giải nén keyframes từ Drive
import zipfile, os
from pathlib import Path

zip_dir = Path('/content/drive/MyDrive/AIC2026')   # thư mục chứa zip trên Drive
out_dir = Path('/content/keyframes')                # giải nén ra ổ đĩa Colab

# Kiểm tra nếu đã unzip từ trước
if out_dir.exists() and len(list(out_dir.iterdir())) > 0:
    print(f'✅ [Skip] /content/keyframes đã có sẵn {len(list(out_dir.iterdir()))} thư mục video.')
else:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(zip_dir.glob('keyframes_L*.zip'))
    print(f'Tìm thấy {len(zip_files)} file zip: {[f.name for f in zip_files]}')

    for zf in zip_files:
        print(f'Đang giải nén {zf.name} ({zf.stat().st_size/1024/1024/1024:.2f} GB) ...')
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(out_dir)
        print(f'  ✅ Xong {zf.name}')

# Tạo symlink liên kết vào thư mục project
project_kf = Path('/content/AIC_2026/data/btc/Keyframes')
project_kf.parent.mkdir(parents=True, exist_ok=True)
if project_kf.is_symlink():
    project_kf.unlink()
if not project_kf.exists():
    os.symlink('/content/keyframes', str(project_kf))

print(f'\n✅ Hoàn tất! Đã liên kết {len(list(project_kf.iterdir()))} thư mục video vào project.')
```

---

#### B.4 Cài EasyOCR & Chạy OCR theo Lesson (Cell 4 & Cell 5)

```python
# Cell 4: Cài EasyOCR & Xem danh sách thống kê Lesson
!pip install easyocr -q
print('✅ Đã cài đặt EasyOCR!')

!cd /content/AIC_2026 && python -m src.indexing.run_ocr_worker --plan
```

```python
# Cell 5: Chạy OCR theo Lesson được phân công
# 🛡️ TỰ ĐỘNG REAL-TIME SYNC VÀO GOOGLE DRIVE sau mỗi video:
#    → Dù Colab bị ngắt kết nối / đổi máy ảo → KHÔNG BAO GIỜ MẤT DỮ LIỆU
# 🔄 TỰ ĐỘNG RESUME: Khởi động lại → script tự nạp tiến độ cũ từ Drive và chạy tiếp
# Ví dụ phân công:
#   Người 1: --lessons L21,L22,L23,L24,L27,L28,L29,L30
#   Người 2: --lessons L25
#   Người 3: --lessons L26
!cd /content/AIC_2026 && python -m src.indexing.run_ocr_worker --lessons L21,L22,L23,L24,L27,L28,L29,L30
```

> ✅ **Không cần Cell 6 nữa!** Code đã tự động đồng bộ vào `MyDrive/AIC2026/ocr_results/` sau **mỗi 1 video** hoàn thành.
> Khi Colab bị ngắt và bạn khởi động lại, chỉ cần chạy lại Cell 1 → Cell 5, script sẽ tự nạp lại từ Drive và làm tiếp phần còn dở.

---

#### B.6 Gộp các file OCR lại (Merge)

Mỗi người tải các file `ocr_part_*.json` từ Drive về bỏ vào thư mục `data/processed/` trên máy tính của mình, rồi mở PowerShell gõ:

```powershell
python -m src.indexing.run_ocr_worker --merge
```

→ Hệ thống sẽ tự động quét tất cả các file `ocr_part_*.json` và gộp thành `data/processed/ocr_index.json` sẵn sàng dùng cho cả nhóm!

---

### ⏱️ Thống kê thời gian thực tế theo từng Lesson

| Lesson | Số Video | Số Ảnh | GPU Colab T4 | CPU Local | Gợi ý phân công |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **L21** | 29 | 7,800 | ~52 phút | ~3.2h | **Người 1** (`--lessons L21,L22,L23,L24,L27,L28,L29,L30`) |
| **L22** | 31 | 9,096 | ~1.0h | ~3.8h | |
| **L23** | 25 | 2,326 | ~16 phút | ~1.0h | |
| **L24** | 43 | 6,781 | ~45 phút | ~2.8h | |
| **L25** | 88 | 37,445 | **~4.2h** | ~15.6h | **Người 2** (`--lessons L25`) |
| **L26** | 497 | 79,428 | **~8.8h** | ~33.1h | **Người 3** (`--lessons L26`) |
| **L27** | 16 | 4,914 | ~33 phút | ~2.0h | |
| **L28** | 24 | 10,683 | ~1.2h | ~4.5h | |
| **L29** | 23 | 10,771 | ~1.2h | ~4.5h | |
| **L30** | 96 | 7,915 | ~53 phút | ~3.3h | |
| **TỔNG** | **872** | **177,159** | **~19.7h** | **~73.8h** | |

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
