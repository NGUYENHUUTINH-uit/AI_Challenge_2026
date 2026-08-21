"""
retriever.py \u2014 Task A (Search): T\u00ecm ki\u1ebfm keyframe b\u1eb1ng CLIP + FAISS
====================================================================
K\u1ebft n\u1ed1i th\u1eadt v\u1edbi:
  - FAISS index (index_self.faiss ho\u1eb7c index_clean.faiss)
  - frame_index_self.json  (mapping faiss_idx \u2192 video_id, timestamp, filename)
  - CLIP model (encode text query \u2192 vector 512 chi\u1ec1u)
  - objects/*.json (YOLOv8m output, d\u00f9ng \u0111\u1ec3 re-rank)
"""

import json
import numpy as np
from pathlib import Path

# ============================================================
# \u0110\u01af\u1edcNG D\u1eaaN
# ============================================================
SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
BTC_DIR      = PROJECT_ROOT / "data" / "btc"

# Ưu tiên BTC index → self → clean
_SOURCES = [
    {
        "faiss":     DATA_DIR / "index_btc.faiss",
        "map":       DATA_DIR / "frame_index_btc.json",
        "keyframes": BTC_DIR  / "Keyframes",
    },
    {
        "faiss":     DATA_DIR / "index_self.faiss",
        "map":       DATA_DIR / "frame_index_self.json",
        "keyframes": DATA_DIR / "keyframes",
    },
    {
        "faiss":     DATA_DIR / "index_clean.faiss",
        "map":       DATA_DIR / "frame_index_clean.json",
        "keyframes": DATA_DIR / "keyframes_clean",
    },
]

OBJECTS_DIR     = DATA_DIR / "objects"           # nhóm tự detect (YOLOv8m COCO)
BTC_OBJECTS_DIR = BTC_DIR  / "Objects"           # BTC cấp (Faster R-CNN Open Images)


def _is_vietnamese(text: str) -> bool:
    """Phát hiện tiếng Việt bằng cách kiểm tra các dấu thanh đặc trưng."""
    import unicodedata

    if not isinstance(text, str):
        return False
    
    for ch in text:
        name = unicodedata.name(ch, "")
        # Tiếng Việt có combining tone marks (COMBINING GRAVE/ACUTE/HOOK/TILDE/DOT)
        # và các chữ cái đặc trưng như Đ/đ, Ơ/ơ, Ư/ư, Ă/ă
        if any(x in name for x in (
            "COMBINING", "WITH GRAVE", "WITH ACUTE", "WITH HOOK",
            "WITH TILDE", "WITH DOT BELOW", "WITH CIRCUMFLEX",
        )):
            return True
        if ch in "ĐđƠơƯưĂă":
            return True
    return False


_TRANS_CACHE: dict = {}


def _translate_vi_to_en(text: str) -> str:
    """
    Dịch tiếng Việt → tiếng Anh chính xác và trung thực (Literal translation).
    """
    if not text:
        return ""
    text_clean = text.strip()
    if text_clean in _TRANS_CACHE:
        return _TRANS_CACHE[text_clean]

    import os
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            r = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=f"Translate the following Vietnamese text to English literally and accurately for video search. Return ONLY the direct English translation without extra words or quotes:\n\"{text_clean}\""
            )
            trans = r.text.strip().strip('"').strip("'").strip("`")
            if trans and len(trans) > 1:
                _TRANS_CACHE[text_clean] = trans
                return trans
        except Exception:
            pass

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source='vi', target='en').translate(text_clean)
        _TRANS_CACHE[text_clean] = translated
        return translated
    except Exception:
        return text_clean

class Retriever:
    """
    Retriever th\u1eadt \u2014 n\u1ed1i v\u00e0o FAISS + CLIP + YOLOv8m objects.

    S\u1eed d\u1ee5ng:
        r = Retriever()
        results = r.search_keyframes("ng\u01b0\u1eddi m\u1eb7c \u00e1o \u0111\u1ecf", top_k=10)
    """

    def __init__(self):
        self._index        = None
        self._frame_map    = None
        self._model        = None
        self._preprocess   = None
        self._device       = None
        self._backend      = None
        self._keyframe_dir = None
        self._loaded       = False

    # ----------------------------------------------------------
    # LAZY LOAD \u2014 ch\u1ec9 load khi c\u00f3 tr\u1ef9 v\u1ea5n \u0111\u1ea7u ti\u00ean
    # ----------------------------------------------------------
    def _ensure_loaded(self):
        if self._loaded:
            return

        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Load FAISS index (th\u1eed t\u1eebng source theo \u01b0u ti\u00ean)
        import faiss
        for src in _SOURCES:
            if src["faiss"].exists() and src["map"].exists():
                print(f"[Retriever] Loading FAISS from {src['faiss'].name} ...")
                self._index = faiss.read_index(str(src["faiss"]))
                with open(src["map"], "r", encoding="utf-8") as f:
                    self._frame_map = json.load(f)
                self._keyframe_dir = src["keyframes"]
                print(f"[Retriever] OK \u2014 {self._index.ntotal} vectors, dim={self._index.d}")
                break

        if self._index is None:
            raise FileNotFoundError(
                "Kh\u00f4ng t\u00ecm th\u1ea5y FAISS index. H\u00e3y ch\u1ea1y:\n"
                "  python run.py --stage build-self\n"
                "(Ho\u1eb7c gi\u1ea3i n\u00e9n colab_results.zip v\u00e0o data/processed/)"
            )

        # 2. Load CLIP model
        print(f"[Retriever] Loading CLIP model on {self._device.upper()} ...")
        try:
            import clip
            self._model, self._preprocess = clip.load("ViT-B/32", device=self._device)
            self._model.eval()
            self._backend = "openai"
            print("[Retriever] OK \u2014 openai/CLIP ViT-B/32")
        except ImportError:
            from transformers import CLIPProcessor, CLIPModel
            self._model      = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self._device)
            self._preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._model.eval()
            self._backend = "huggingface"
            print("[Retriever] OK \u2014 HuggingFace CLIP ViT-B/32 (fallback)")

        self._loaded = True

    # ----------------------------------------------------------
    # ENCODE TEXT QUERY → VECTOR
    # ----------------------------------------------------------
    def _encode_text(self, query: str) -> np.ndarray:
        """
        Encode câu text → vector 512 chiều (float32, L2-normalized).
        Tự động dịch Việt → Anh trước khi encode vì CLIP train bằng tiếng Anh.
        """
        # Dịch sang tiếng Anh nếu phát hiện tiếng Việt
        if _is_vietnamese(query):
            query = _translate_vi_to_en(query)

        import torch
        with torch.no_grad():
            if self._backend == "openai":
                import clip
                tokens = clip.tokenize([query], truncate=True).to(self._device)
                feat   = self._model.encode_text(tokens)
            else:
                inputs = self._preprocess(
                    text=[query], return_tensors="pt",
                    padding=True, truncation=True
                ).to(self._device)
                
                # Sửa tại đây: Gọi thẳng hàm lấy text features của mô hình HuggingFace CLIP
                feat = self._model.get_text_features(**inputs)
                
                # Phòng hờ nếu trả về object bọc bên ngoài
                if hasattr(feat, "pooler_output"):
                    feat = feat.pooler_output
                elif hasattr(feat, "text_embeds"):
                    feat = feat.text_embeds

            feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 normalize
        return feat.cpu().numpy().astype(np.float32)

    # ----------------------------------------------------------
    # ĐỌC OBJECTS — hỗ trợ 2 format: BTC và nhóm (YOLOv8m)
    # ----------------------------------------------------------
    def _load_objects(self, video_id: str, frame_name: str,
                      n: int = None,
                      score_threshold: float = 0.25) -> list:
        """
        Đọc objects theo thứ tự ưu tiên:
          1. BTC format (data/btc/Objects/<video_id>/<n:03d>.json)
             Field: detection_class_entities + detection_scores
          2. Nhóm format (data/processed/objects/<video_id>/<stem>.json)
             Field: [{"class": ..., "score": ...}]
        Trả về list tên class, ví dụ: ["person", "car", "bicycle"]
        """
        # --- Thử BTC format trước (ưu tiên cao hơn) ---
        if n is not None:
            btc_path = BTC_OBJECTS_DIR / video_id / f"{int(n):03d}.json"
            if btc_path.exists():
                try:
                    with open(btc_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    entities = data.get("detection_class_entities", [])
                    scores   = data.get("detection_scores", [])
                    # Lọc theo score threshold, trả về tên class
                    classes = [
                        ent for ent, sc in zip(entities, scores)
                        if float(sc) >= score_threshold
                    ]
                    if classes:
                        return classes
                except Exception:
                    pass

        # --- Fallback: nhóm tự detect (YOLOv8m COCO) ---
        stem     = Path(frame_name).stem
        obj_path = OBJECTS_DIR / video_id / f"{stem}.json"
        if not obj_path.exists():
            return []
        try:
            with open(obj_path, "r", encoding="utf-8") as f:
                detections = json.load(f)
            return [d["class"] for d in detections if "class" in d]
        except Exception:
            return []

    # ----------------------------------------------------------
    # SEARCH CH\u00cdNH
    # ----------------------------------------------------------
    def search_keyframes(self,
                         query_text: str = None,
                         query_image=None,
                         top_k: int = 20,
                         video_filter: str = None) -> list:
        """
        Tìm kiếm keyframe bằng text hoặc ảnh.
        Có hỗ trợ video_filter để chỉ tìm trong 1 video cụ thể.
        """
        self._ensure_loaded()

        if query_text is None and query_image is None:
            return []

        # 1. Encode query → vector
        if query_text:
            query_vec = self._encode_text(query_text)    # (1, 512)
        else:
            return []

        # 2. FAISS search (nếu có filter video thì lấy nhiều hơn để lọc)
        fetch_k = top_k * 5 if video_filter else top_k
        scores, indices = self._index.search(query_vec, min(fetch_k, self._index.ntotal))
        scores  = scores[0].tolist()
        indices = indices[0].tolist()

        # 3. Map FAISS index → thông tin frame
        results = []
        for score, faiss_idx in zip(scores, indices):
            if faiss_idx < 0:
                continue

            info = self._frame_map.get(str(faiss_idx), {})

            filename = info.get("filename")
            if not filename:
                n_val = info.get("n")
                if n_val is not None:
                    filename = f"{int(n_val):03d}.jpg"
                else:
                    frame_idx_val = info.get("frame_idx", 0)
                    filename = f"{int(frame_idx_val):04d}.jpg"

            video_id = info.get("video_id") or (
                filename.split("/")[0] if "/" in filename else "unknown"
            )

            # Lọc theo video nếu có yêu cầu
            if video_filter and video_id.strip().upper() != video_filter.strip().upper():
                continue

            frame_name = Path(filename).name

            timestamp = info.get("timestamp")
            if timestamp is None:
                timestamp = info.get("pts_time")
            if timestamp is not None:
                timestamp = round(float(timestamp), 2)

            frame_id = info.get("frame_id")
            if frame_id is None:
                frame_id = info.get("frame_idx")
            if frame_id is None:
                n_raw = info.get("n")
                frame_id = int(n_raw) - 1 if n_raw is not None else 0
            frame_id = int(frame_id)

            n_val = info.get("n")
            objects = self._load_objects(
                video_id, frame_name,
                n=int(n_val) if n_val is not None else None
            )

            from src.indexing.ocr_indexer import get_ocr_indexer
            ocr_text = get_ocr_indexer().get_text(video_id, frame_name)

            img_path = str(self._keyframe_dir / video_id / frame_name)

            results.append({
                "video_id":   video_id,
                "filename":   frame_name,
                "timestamp":  timestamp,
                "frame_id":   frame_id,
                "score":      round(float(score), 4),
                "objects":    objects,
                "ocr_text":   ocr_text,
                "image_path": img_path,
                "scene_id":   info.get("scene_id"),
            })

            if len(results) >= top_k:
                break

        return results

    def get_video_timeline(self, video_id: str) -> list:
        """
        Lấy toàn bộ danh sách keyframe của một video theo thứ tự thời gian (Chronological timeline).
        Dùng cho tính năng Drill-down / Xem toàn bộ frame của video.
        """
        self._ensure_loaded()
        v_clean = video_id.strip().upper()
        frames = []

        for faiss_idx_str, info in self._frame_map.items():
            vid = (info.get("video_id") or "").strip().upper()
            if vid == v_clean:
                filename = info.get("filename")
                if not filename:
                    n_val = info.get("n")
                    if n_val is not None:
                        filename = f"{int(n_val):03d}.jpg"
                    else:
                        frame_idx_val = info.get("frame_idx", 0)
                        filename = f"{int(frame_idx_val):04d}.jpg"

                frame_name = Path(filename).name

                timestamp = info.get("timestamp")
                if timestamp is None:
                    timestamp = info.get("pts_time")
                if timestamp is not None:
                    timestamp = round(float(timestamp), 2)

                frame_id = info.get("frame_id")
                if frame_id is None:
                    frame_id = info.get("frame_idx")
                if frame_id is None:
                    n_raw = info.get("n")
                    frame_id = int(n_raw) - 1 if n_raw is not None else 0
                frame_id = int(frame_id)

                n_val = info.get("n")
                objects = self._load_objects(
                    video_id, frame_name,
                    n=int(n_val) if n_val is not None else None
                )

                from src.indexing.ocr_indexer import get_ocr_indexer
                ocr_text = get_ocr_indexer().get_text(video_id, frame_name)
                img_path = str(self._keyframe_dir / video_id / frame_name)

                frames.append({
                    "video_id":   video_id,
                    "filename":   frame_name,
                    "timestamp":  timestamp or 0.0,
                    "frame_id":   frame_id,
                    "objects":    objects,
                    "ocr_text":   ocr_text,
                    "image_path": img_path,
                    "n":          int(n_val) if n_val is not None else 0
                })

        # Sắp xếp theo timestamp hoặc số thứ tự n
        frames.sort(key=lambda x: (x["timestamp"], x["frame_id"], x["n"]))
        return frames


if __name__ == "__main__":
    # Test nhanh
    r = Retriever()
    res = r.search_keyframes("a person riding a motorcycle", top_k=5)
    for item in res:
        print(f"{item['video_id']} | {item['filename']} | score={item['score']:.4f} | objects={item['objects']}")

