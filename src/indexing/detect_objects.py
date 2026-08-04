"""
detect_objects.py — Object Detection bang Faster R-CNN (torchvision)
=====================================================================
Chay Faster R-CNN tren tung keyframe de tao file JSON object detection.
Output tuong thich voi load_btc_data.load_objects_for_video().

BTC dung: Faster R-CNN pretrained tren OpenImages V4 (600 classes)
Minh dung: Faster R-CNN pretrained tren COCO (80 classes) - don gian hon,
           du dung cho rerank_by_objects() trong thi dau.

Cau truc output:
  data/processed/objects/<video_id>/0000.json
  data/processed/objects/<video_id>/0001.json
  ...

Moi file JSON:
  [
    {"class": "person",  "score": 0.98, "box": [x1,y1,x2,y2]},
    {"class": "bicycle", "score": 0.87, "box": [x1,y1,x2,y2]}
  ]

Chay:
  # Toan bo video da trich keyframe:
  python -m src.indexing.detect_objects

  # 1 video cu the:
  python -m src.indexing.detect_objects --video-id L21_V001

  # Chi dinh nguon keyframe:
  python -m src.indexing.detect_objects --keyframe-dir data/btc/Keyframes
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import argparse
import torch
import torchvision
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ============================================================
# COCO CLASS NAMES (80 classes) — dung cho model mac dinh
# ============================================================
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Grounding DINO: danh sach class cho cac video surveillance/sousveillance
# Them/bot class o day de tuy chinh cho tung loai video AIC 2026
GROUNDING_DINO_CLASSES = [
    # Nguoi
    "person", "man", "woman", "child", "crowd",
    # Phuong tien
    "car", "motorcycle", "bicycle", "bus", "truck", "van", "taxi",
    "scooter", "ambulance", "police car",
    # Duong pho
    "traffic light", "stop sign", "road sign", "crosswalk", "sidewalk",
    # Do vat cam tay
    "phone", "bag", "backpack", "umbrella", "helmet", "hat",
    # Noi that / cong trinh
    "building", "door", "window", "chair", "table", "bench", "stairs",
    # Dong vat
    "dog", "cat", "bird",
    # Hanh dong / vat the khac
    "laptop", "bottle", "cup", "fire", "smoke", "tree", "fence",
]


# ============================================================
# DUONG DAN
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
PROCESSED    = DATA_DIR / "processed"

# Nguon keyframe mac dinh: output cua ingest.py
KF_DIR_SELF = PROCESSED / "keyframes"         # data/processed/keyframes/<video_id>/
KF_DIR_BTC  = DATA_DIR  / "btc" / "Keyframes" # data/btc/Keyframes/<video_id>/

# Output objects
OBJ_OUTPUT  = PROCESSED / "objects"           # data/processed/objects/<video_id>/

# 🚀 Batch size cho object detection
DETECT_BATCH_SIZE = 8

# File metadata_clean (output cua preprocess + group_scenes)
METADATA_CLEAN = PROCESSED / "metadata_clean.json"


# ============================================================
# HELPER: LOAD REPRESENTATIVE FRAMES
# ============================================================

def _load_representative_set(metadata_path: Path = METADATA_CLEAN) -> set:
    """
    🚀 Đọc metadata_clean.json, trả về set các key ("video_id/filename.jpg")
    có is_representative=True.

    Dùng để detect object chỉ trên frame đại diện thay vì toàn bộ.
    Giảm số frame detect ~60-80% → nhanh hơn ~3-5x.

    Returns:
        set of "video_id/filename.jpg" strings (rỗng nếu không có metadata)
    """
    if not metadata_path.exists():
        print(f"  [WARN] Không tìm thấy {metadata_path.name}")
        print(f"         Hãy chạy: python run.py --stage filter && python run.py --stage group-scenes")
        print(f"         Fallback: detect toàn bộ frame.")
        return set()

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rep_set = {
        key for key, info in meta.items()
        if info.get("is_representative", False)
    }
    print(f"  [REP] Representative frames: {len(rep_set)} / {len(meta)} total "
          f"({100*len(rep_set)/max(len(meta),1):.1f}%)")
    return rep_set


# ============================================================
# LOAD MODEL — FACTORY (chon theo --model)
# ============================================================

def load_model(model_name: str = "coco", device: str = None):
    """
    Factory function: tra ve (model, device, model_name).

    model_name:
      'coco'          : Faster R-CNN torchvision COCO 80 classes  [mac dinh, khong can cai them]
      'openimages'    : Faster R-CNN TF Hub OpenImages V4 600 classes [giong het BTC, can tensorflow]
      'grounding-dino': Grounding DINO zero-shot [linh hoat nhat, can pip install groundingdino]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "openimages":
        return _load_openimages(device)
    elif model_name == "grounding-dino":
        return _load_grounding_dino(device)
    else:
        return _load_coco(device)


def _load_coco(device: str):
    """Faster R-CNN torchvision — COCO 80 classes."""
    print(f"[MODEL] Faster R-CNN COCO (80 classes) on {device.upper()}")
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    print("[MODEL] OK")
    return model, device, "coco"


def _load_openimages(device: str):
    """
    Faster R-CNN TF Hub — OpenImages V4 (600 classes) — GIONG HET BTC.
    Can: pip install tensorflow tensorflow-hub
    """
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError:
        raise ImportError(
            "Can cai tensorflow va tensorflow-hub:\n"
            "  pip install tensorflow tensorflow-hub"
        )

    print(f"[MODEL] Loading Faster R-CNN OpenImages V4 (600 classes) via TF Hub...")
    # Model URL chinh thuc tu TF Hub (giong BTC su dung)
    MODULE_HANDLE = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.load(MODULE_HANDLE)
    print("[MODEL] OK — OpenImages V4 600 classes")
    return detector, device, "openimages"


def _load_grounding_dino(device: str):
    """
    Grounding DINO — zero-shot object detection, khong gioi han class.
    Can: pip install groundingdino-py
    """
    try:
        from groundingdino.util.inference import load_model as gd_load
        from groundingdino.util.inference import predict
    except ImportError:
        raise ImportError(
            "Can cai groundingdino:\n"
            "  pip install groundingdino-py"
        )

    print(f"[MODEL] Loading Grounding DINO (zero-shot) on {device.upper()}...")
    # Tai config va weights tu HuggingFace
    import urllib.request, tempfile
    cfg_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    wt_url  = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"

    tmp = tempfile.mkdtemp()
    cfg_path = f"{tmp}/config.py"
    wt_path  = f"{tmp}/weights.pth"
    urllib.request.urlretrieve(cfg_url, cfg_path)
    urllib.request.urlretrieve(wt_url,  wt_path)

    model = gd_load(cfg_path, wt_path)
    model = model.to(device)
    print("[MODEL] OK — Grounding DINO zero-shot")
    return model, device, "grounding-dino"



# ============================================================
# DETECT 1 FRAME (ho tro ca 3 backend)
# ============================================================

def detect_frame(image_path: Path, model, device: str,
                 model_name: str = "coco",
                 score_threshold: float = 0.3) -> list:
    """
    Chay object detection tren 1 anh, tu dong chon backend dua theo model_name.
    Tra ve list: [{"class": "person", "score": 0.98, "box": [x1,y1,x2,y2]}, ...]
    """
    if model_name == "openimages":
        return _detect_openimages(image_path, model, score_threshold)
    elif model_name == "grounding-dino":
        return _detect_grounding_dino(image_path, model, device, score_threshold)
    else:
        return _detect_coco(image_path, model, device, score_threshold)


def _detect_coco(image_path, model, device, score_threshold):
    """COCO 80 classes — detect 1 frame (wrapper gọi batch version)."""
    results_list = _detect_coco_batch([image_path], model, device, score_threshold)
    return results_list[0] if results_list else []


def _detect_coco_batch(image_paths: list, model, device: str,
                       score_threshold: float = 0.3) -> list:
    """
    🚀 COCO 80 classes — detect NHIỀU FRAME trong 1 forward pass.
    Faster R-CNN torchvision hỗ trợ batch natively: model([t1, t2, ...]).
    Dùng fp16 autocast trên GPU → nhanh hơn ~2-3x so với frame-by-frame.

    Args:
        image_paths: list[Path] — danh sách frame cần detect
    Returns:
        list[list[dict]] — mỗi phần tử tương ứng với 1 frame trong image_paths
    """
    import torchvision.transforms.functional as F

    tensors = []
    valid_indices = []  # index của frame load thành công
    results_all = [[] for _ in range(len(image_paths))]  # pre-fill với list rỗng

    for i, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            tensors.append(F.to_tensor(img).to(device))
            valid_indices.append(i)
        except Exception as e:
            print(f"  [WARN] {img_path.name}: {e}")

    if not tensors:
        return results_all

    # fp16 autocast trên GPU → nhanh hơn ~1.5-2x
    use_amp = device != "cpu" and torch.cuda.is_available()
    with torch.no_grad():
        if use_amp:
            with torch.cuda.amp.autocast():
                batch_outputs = model(tensors)
        else:
            batch_outputs = model(tensors)

    for frame_i, output in zip(valid_indices, batch_outputs):
        frame_results = []
        for label, score, box in zip(
            output["labels"].cpu().tolist(),
            output["scores"].cpu().tolist(),
            output["boxes"].cpu().tolist()
        ):
            if score < score_threshold:
                continue
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else str(label)
            frame_results.append({
                "class": class_name,
                "score": round(score, 4),
                "box":   [round(v, 1) for v in box],
            })
        results_all[frame_i] = frame_results

    return results_all


def _detect_openimages(image_path, detector, score_threshold):
    """
    OpenImages V4 600 classes — dung TF Hub.
    Output giong het format BTC.
    """
    import tensorflow as tf
    import numpy as np
    img = Image.open(image_path).convert("RGB")
    img_tensor = tf.constant(np.array(img))[tf.newaxis, ...]
    result = detector(img_tensor)
    result = {k: v.numpy() for k, v in result.items()}

    results = []
    n = len(result["detection_scores"][0])
    for i in range(n):
        score = float(result["detection_scores"][0][i])
        if score < score_threshold:
            break  # da sorted giam dan
        class_name = result["detection_class_entities"][0][i].decode("utf-8")
        box = result["detection_boxes"][0][i].tolist()   # [ymin, xmin, ymax, xmax]
        results.append({
            "class": class_name,
            "score": round(score, 4),
            "box":   [round(v, 4) for v in box],
        })
    return results


def _detect_grounding_dino(image_path, model, device, score_threshold):
    """
    Grounding DINO zero-shot — dung GROUNDING_DINO_CLASSES lam prompt.
    """
    from groundingdino.util.inference import predict
    import torchvision.transforms as T

    img = Image.open(image_path).convert("RGB")
    transform = T.Compose([
        T.Resize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).to(device)
    prompt = " . ".join(GROUNDING_DINO_CLASSES) + " ."

    boxes, logits, phrases = predict(
        model=model, image=img_tensor,
        caption=prompt,
        box_threshold=score_threshold,
        text_threshold=score_threshold,
    )
    results = []
    for phrase, score in zip(phrases, logits.tolist()):
        results.append({
            "class": phrase,
            "score": round(score, 4),
        })
    return results


# ============================================================
# DETECT 1 VIDEO
# ============================================================

def detect_video(video_id: str, model, device: str,
                 model_name: str = "coco",
                 keyframe_dir: Path = KF_DIR_SELF,
                 output_dir: Path = OBJ_OUTPUT,
                 score_threshold: float = 0.3,
                 skip_existing: bool = True,
                 batch_size: int = DETECT_BATCH_SIZE,
                 rep_only_set: set = None) -> int:
    """
    Chay object detection tren tat ca keyframe cua 1 video.
    Luu JSON vao output_dir/<video_id>/<frame_name>.json.

    🚀 Tối ưu: Gom frames thành batch (batch_size=8) thay vì detect từng frame.
    Faster R-CNN torchvision hỗ trợ batch natively → nhanh hơn ~3-5x trên GPU.

    Args:
        rep_only_set: set of "video_id/filename.jpg" — nếu không None,
                      chỉ detect frame có key nằm trong set này.
    Returns: So frame da xu ly
    """
    kf_video_dir = keyframe_dir / video_id
    if not kf_video_dir.exists():
        print(f"  [WARN] Khong tim thay keyframe cho {video_id}: {kf_video_dir}")
        return 0

    jpg_files = sorted(kf_video_dir.glob("*.jpg"))
    if not jpg_files:
        print(f"  [WARN] Khong co file .jpg trong {kf_video_dir}")
        return 0

    # Lọc representative frames nếu có yêu cầu
    if rep_only_set:
        jpg_files = [
            p for p in jpg_files
            if f"{video_id}/{p.name}" in rep_only_set
        ]
        if not jpg_files:
            print(f"  [WARN] Không có representative frame nào cho {video_id} trong metadata_clean.json")
            return 0

    out_video_dir = output_dir / video_id
    out_video_dir.mkdir(parents=True, exist_ok=True)

    # Lọc ra những frame chưa có JSON (skip_existing)
    pending = []
    count = 0
    for jpg_path in jpg_files:
        out_json = out_video_dir / f"{jpg_path.stem}.json"
        if skip_existing and out_json.exists():
            count += 1
        else:
            pending.append(jpg_path)

    if not pending:
        return count

    # Batch processing
    use_batch = (model_name == "coco" and batch_size > 1)

    if use_batch:
        for i in tqdm(range(0, len(pending), batch_size),
                      desc=f"  {video_id}", leave=False):
            batch_paths = pending[i: i + batch_size]
            try:
                batch_results = _detect_coco_batch(
                    batch_paths, model, device, score_threshold
                )
                for jpg_path, detections in zip(batch_paths, batch_results):
                    out_json = out_video_dir / f"{jpg_path.stem}.json"
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(detections, f, ensure_ascii=False)
                    count += 1
            except Exception as e:
                print(f"  [WARN] batch error: {e} — falling back to frame-by-frame")
                for jpg_path in batch_paths:
                    out_json = out_video_dir / f"{jpg_path.stem}.json"
                    try:
                        det = _detect_coco([jpg_path], model, device, score_threshold)
                        with open(out_json, "w", encoding="utf-8") as f:
                            json.dump(det, f, ensure_ascii=False)
                        count += 1
                    except Exception as e2:
                        print(f"  [WARN] {jpg_path.name}: {e2}")
    else:
        for jpg_path in tqdm(pending, desc=f"  {video_id}", leave=False):
            out_json = out_video_dir / f"{jpg_path.stem}.json"
            try:
                detections = detect_frame(jpg_path, model, device,
                                          model_name=model_name,
                                          score_threshold=score_threshold)
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(detections, f, ensure_ascii=False)
                count += 1
            except Exception as e:
                print(f"  [WARN] {jpg_path.name}: {e}")

    return count


# ============================================================
# PIPELINE CHINH
# ============================================================

def run_detect(keyframe_dir: Path = KF_DIR_SELF,
               output_dir: Path = OBJ_OUTPUT,
               video_id: str = None,
               model_name: str = "coco",
               score_threshold: float = 0.3,
               skip_existing: bool = True,
               rep_only: bool = False):
    """Chay object detection toan bo thu muc keyframe.

    Args:
        rep_only: Nếu True, chỉ detect frame có is_representative=True
                  trong metadata_clean.json (cần chạy filter + group-scenes trước).
                  Giảm số frame detect ~60-80% → nhanh hơn ~3-5x.
    """
    model, device, model_name = load_model(model_name)

    # Load representative set nếu có yêu cầu
    rep_only_set = _load_representative_set() if rep_only else None

    if video_id:
        video_dirs = [keyframe_dir / video_id]
    else:
        video_dirs = sorted([d for d in keyframe_dir.iterdir() if d.is_dir()])

    if not video_dirs:
        print(f"Khong tim thay thu muc video trong: {keyframe_dir}")
        return

    model_labels = {
        "coco":           "COCO 80 classes (torchvision)",
        "openimages":     "OpenImages V4 600 classes (TF Hub) — GIONG HET BTC",
        "grounding-dino": f"Grounding DINO zero-shot ({len(GROUNDING_DINO_CLASSES)} custom classes)",
    }

    mode_label = "REPRESENTATIVE ONLY" if rep_only else "ALL FRAMES"
    print(f"\n{'='*60}")
    print(f"  OBJECT DETECTION — {len(video_dirs)} videos  [{mode_label}]")
    print(f"  Model  : {model_labels.get(model_name, model_name)}")
    print(f"  Device : {device.upper()}")
    print(f"  Nguon  : {keyframe_dir}")
    print(f"  Output : {output_dir}")
    print(f"{'='*60}\n")

    total_frames = 0
    for vdir in video_dirs:
        vid = vdir.name
        print(f"[{vid}]")
        n = detect_video(
            video_id=vid,
            model=model,
            device=device,
            model_name=model_name,
            keyframe_dir=keyframe_dir,
            output_dir=output_dir,
            score_threshold=score_threshold,
            skip_existing=skip_existing,
            rep_only_set=rep_only_set,
        )
        print(f"  Done: {n} frames")
        total_frames += n

    print(f"\n{'='*60}")
    print(f"DETECT DONE! {total_frames} frames [{mode_label}]")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Object Detection tren Keyframes",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--model", choices=["coco", "openimages", "grounding-dino"], default="coco",
        help=(
            "Chon model:\n"
            "  coco          : Faster R-CNN COCO 80 classes [mac dinh, khong can cai them]\n"
            "  openimages    : Faster R-CNN OpenImages V4 600 classes [giong het BTC]\n"
            "                  -> can: pip install tensorflow tensorflow-hub\n"
            "  grounding-dino: Zero-shot, dung GROUNDING_DINO_CLASSES [linh hoat nhat]\n"
            "                  -> can: pip install groundingdino-py"
        )
    )
    parser.add_argument(
        "--source", choices=["self", "btc"], default="self",
        help=(
            "Nguon keyframe:\n"
            "  self: data/processed/keyframes/ (output ingest.py) [default]\n"
            "  btc : data/btc/Keyframes/ (BTC cung cap)"
        )
    )
    parser.add_argument(
        "--keyframe-dir", type=str, default=None,
        help="Tuy chinh thu muc keyframe (ghi de --source)"
    )
    parser.add_argument(
        "--video-id", type=str, default=None,
        help="Chi xu ly 1 video cu the (vi du: L21_V001). Mac dinh: tat ca"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.3,
        help="Nguong confidence toi thieu (default: 0.3)"
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Chay lai du da co file JSON (mac dinh: bo qua file da xu ly)"
    )
    parser.add_argument(
        "--rep-only", action="store_true",
        help=(
            "🚀 Chỉ detect frame đại diện (is_representative=True từ metadata_clean.json).\n"
            "   Giảm số frame detect ~60-80%% → nhanh hơn ~3-5x.\n"
            "   Yêu cầu chạy trước: python run.py --stage filter && python run.py --stage group-scenes"
        )
    )
    args = parser.parse_args()

    if args.keyframe_dir:
        kf_dir = Path(args.keyframe_dir)
    elif args.source == "btc":
        kf_dir = KF_DIR_BTC
    else:
        kf_dir = KF_DIR_SELF

    run_detect(
        keyframe_dir=kf_dir,
        output_dir=OBJ_OUTPUT,
        video_id=args.video_id,
        model_name=args.model,
        score_threshold=args.score_threshold,
        skip_existing=not args.no_skip,
        rep_only=args.rep_only,
    )
