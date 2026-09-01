"""
ocr_indexer.py — Trích xuất và lập chỉ mục OCR (Optical Character Recognition) cho AIC 2026
=========================================================================================
Hỗ trợ:
1. Chạy OCR trên tập keyframes (dùng EasyOCR nếu có, fallback heuristic / mock text).
2. Lưu & tải file index OCR: data/processed/ocr_index.json
3. Tìm kiếm chuỗi ký tự / số / từ khóa trong cơ sở dữ liệu OCR với Fuzzy / Token Match.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
BTC_KEYFRAMES_DIR = DATA_DIR / "btc" / "Keyframes"
OCR_INDEX_FILE = PROCESSED_DIR / "ocr_index.json"


def normalize_text(text: str) -> str:
    """Chuẩn hóa chuỗi văn bản: lowercase, xóa khoảng trắng thừa."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text



# Từ khóa domain đặc thù — thường xuất hiện trực tiếp trên slide/bảng
_DOMAIN_KEYWORDS: list[tuple[str, str]] = [
    # Địa lý / Hành chính
    ("mạng lưới đô thị", "mạng lưới đô thị"),
    ("đô thị", "đô thị"),
    ("dân số đô thị", "dân số đô thị"),
    ("tỉ lệ đô thị hóa", "tỉ lệ đô thị hóa"),
    ("đô thị hóa", "đô thị hóa"),
    ("phân bố đô thị", "phân bố đô thị"),
    ("vùng kinh tế", "vùng kinh tế"),
    ("đồng bằng sông cửu long", "đồng bằng sông cửu long"),
    ("đồng bằng sông hồng", "đồng bằng sông hồng"),
    ("đông nam bộ", "đông nam bộ"),
    ("tây nguyên", "tây nguyên"),
    ("bắc trung bộ", "bắc trung bộ"),
    ("nam trung bộ", "nam trung bộ"),
    ("trung du miền núi", "trung du miền núi"),
    # Giáo dục / Bài giảng
    ("địa lí", "địa lí"), ("địa lý", "địa lý"),
    ("bài giảng", "bài giảng"), ("bài học", "bài học"),
    ("bảng số liệu", "bảng số liệu"), ("số liệu", "số liệu"),
    ("biểu đồ", "biểu đồ"), ("bản đồ", "bản đồ"),
    # Kinh tế / Xã hội
    ("gdp", "gdp"), ("grdp", "grdp"),
    ("dân số", "dân số"), ("mật độ", "mật độ"),
    ("tốc độ tăng trưởng", "tốc độ tăng trưởng"),
    ("xuất khẩu", "xuất khẩu"), ("nhập khẩu", "nhập khẩu"),
    ("việt nam", "việt nam"), ("viet nam", "viet nam"),
    # Lịch sử / Thể thao
    ("huy chương", "huy chương"), ("vô địch", "vô địch"),
    ("sea games", "sea games"), ("asean", "asean"),
    ("tổng bí thư", "tổng bí thư"), ("chủ tịch", "chủ tịch"),
]


def extract_potential_ocr_keywords(query: str) -> List[str]:
    """
    Trích xuất các từ khóa đặc thù có khả năng cao xuất hiện dưới dạng chữ/số trên ảnh:
    - Số hiệu (vd: 54, 12, 100, 2026)
    - Tên viết hoa / Thương hiệu / Tên riêng (vd: KFC, VinFast, VTV, BIDV, KOI)
    - Cụm từ đi kèm 'chữ', 'số', 'biển', 'bảng'
    - Domain keywords: địa lý, giáo dục, kinh tế, xã hội (tiếng Việt)
    """
    if not query:
        return []

    keywords = []
    q_lower = query.lower()

    # 1. Domain keywords — ưu tiên cao nhất (cụm từ dài trước)
    for vi_kw, en_kw in sorted(_DOMAIN_KEYWORDS, key=lambda x: -len(x[0])):
        if vi_kw in q_lower:
            keywords.append(vi_kw)

    # 2. Tìm các chuỗi số (số tuyến bus, số áo, số nhà...)
    numbers = re.findall(r"\b\d+\b", query)
    # Chỉ lấy số có nghĩa (>= 2 chữ số hoặc số >= 10 để tránh match 1, 2, 3 quá chung)
    keywords.extend([n for n in numbers if len(n) >= 2 or int(n) >= 10])

    # 3. Tìm các từ viết hoa hoặc từ viết tắt trong câu gốc
    caps = re.findall(r"\b[A-Z0-9]{2,}\b", query)
    keywords.extend(caps)

    # 4. Tìm các cụm trong dấu ngoặc kép hoặc sau từ 'chữ', 'số', 'biển'
    quotes = re.findall(r'["\']([^"\']+)["\']', query)
    keywords.extend(quotes)

    patterns = [
        r"(?:chữ|từ|biển|bảng|hiệu|số)\\s+([A-Za-z0-9\\s]+?)(?:[,\\.\\?]|$)",
    ]
    for p in patterns:
        matches = re.findall(p, query, flags=re.IGNORECASE)
        for m in matches:
            cleaned = m.strip()
            if cleaned and len(cleaned) <= 30:
                keywords.append(cleaned)

    # Khử trùng lặp và lọc từ quá ngắn
    result = []
    seen = set()
    for kw in keywords:
        norm = normalize_text(kw)
        if norm and len(norm) >= 2 and norm not in seen:
            seen.add(norm)
            result.append(norm)

    return result



class OCRIndexer:
    def __init__(self, index_file: Path = OCR_INDEX_FILE):
        self.index_file = Path(index_file)
        self.ocr_data: Dict[str, Dict[str, str]] = {}  # {video_id: {frame_name: ocr_text}}
        self.loaded = False

    def load_index(self) -> bool:
        """Nạp dữ liệu OCR từ JSON đã lưu."""
        if not self.index_file.exists():
            self.loaded = True
            return False

        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.ocr_data = json.load(f)
            self.loaded = True
            return True
        except Exception as e:
            print(f"⚠️ Không thể đọc file OCR index: {e}")
            self.ocr_data = {}
            self.loaded = True
            return False

    def save_index(self):
        """Lưu cơ sở dữ liệu OCR ra file JSON."""
        self.index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.ocr_data, f, ensure_ascii=False, indent=2)

    def get_text(self, video_id: str, frame_name: str) -> str:
        """Lấy chuỗi OCR của một keyframe cụ thể."""
        if not self.loaded:
            self.load_index()
        stem = Path(frame_name).stem
        video_dict = self.ocr_data.get(video_id, {})
        return video_dict.get(frame_name) or video_dict.get(stem) or ""

    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        """
        Tìm kiếm các frame có chứa text OCR khớp với query.
        Trả về danh sách: [{"video_id": ..., "frame_name": ..., "ocr_score": ..., "text": ...}]
        """
        if not self.loaded:
            self.load_index()

        keywords = extract_potential_ocr_keywords(query)
        if not keywords:
            # Fallback nếu không bóc tách được từ khóa cụ thể: tìm từng từ trong query
            norm_q = normalize_text(query)
            tokens = [t for t in norm_q.split() if len(t) > 2]
            keywords = tokens[:5]

        if not keywords:
            return []

        hits = []
        for video_id, frames in self.ocr_data.items():
            for frame_name, raw_text in frames.items():
                if not raw_text:
                    continue
                norm_text = normalize_text(raw_text)
                matched_count = 0
                exact_match = False

                for kw in keywords:
                    if kw in norm_text:
                        matched_count += 1
                        if f" {kw} " in f" {norm_text} " or kw == norm_text:
                            exact_match = True

                if matched_count > 0:
                    score = 0.5 * (matched_count / len(keywords)) + (0.5 if exact_match else 0.0)
                    hits.append({
                        "video_id": video_id,
                        "filename": frame_name if "." in frame_name else f"{frame_name}.jpg",
                        "ocr_score": round(score, 4),
                        "ocr_text": raw_text,
                    })

        hits.sort(key=lambda x: -x["ocr_score"])
        return hits[:top_k]


# Singleton
_ocr_indexer = OCRIndexer()

def get_ocr_indexer() -> OCRIndexer:
    global _ocr_indexer
    if not _ocr_indexer.loaded:
        _ocr_indexer.load_index()
    return _ocr_indexer
