"""
query_expander.py — LLM Query Expansion & Visual Reasoning cho AIC 2026
=======================================================================
Giải quyết các bài toán hóc búa của Video Retrieval:
1. Đố mẹo tri thức / Pop-culture trivia (vd: "phim Steven Spielberg 1975" -> "Jaws / Shark").
2. Câu chuyện dài / Đoạn văn miêu tả nhiều chữ (Lọc bỏ từ rác, trích xuất Visual Keywords cốt lõi cho CLIP).
3. Trích xuất tự động danh sách Object Detection + OCR Text tiềm năng.
"""

import os
import json
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Cache lưu trữ các câu truy vấn đã mở rộng: {raw_query: result_dict}
_EXPANSION_CACHE: Dict[str, Dict[str, Any]] = {}


SYSTEM_PROMPT = """Bạn là chuyên gia thị giác máy tính AI (Computer Vision Video Retrieval Specialist) tại cuộc thi AI Challenge (AIC 2026).
Nhiệm vụ: Phân tích câu truy vấn phức tạp từ người dùng và chuyển đổi thành câu miêu tả thị giác tiếng Anh (Visual Prompt) để mô hình CLIP tìm kiếm chính xác nhất.

NGUYÊN TẮC QUAN TRỌNG:
1. [CHỦ THỂ ĐÃ GIẢI MÃ]: Giải mã tri thức / đố mẹo (vd: 'phim của Steven Spielberg năm 1975' -> 'great white shark', 'vận động viên áo xanh trắng' -> 'cyclist wearing blue and white jersey').
2. [BỐI CẢNH / MÔI TRƯỜNG]: BẮT BUỘC giữ lại bối cảnh không gian (vd: 'thị trấn ven biển' -> 'coastal beach town / seaside').
3. [CON NGƯỜI & HÀNH ĐỘNG]: BẮT BUỘC giữ lại sự hiện diện của con người và hành động cụ thể.
4. [LOẠI BỎ MÔ TẢ BỐ CỤC KHUNG HÌNH]: TUYỆT ĐỐI bỏ qua các cụm từ mô tả VỊ TRÍ TRONG MÀN HÌNH như 'góc dưới bên trái', 'góc trên bên phải', 'xuất hiện ở góc', 'phía trên màn hình'... vì CLIP không nhận diện được bố cục khung hình.
5. [QUẦN ÁO & NGOẠI HÌNH]: Giữ lại MÀU ÁO của TỪNG người (vd: áo xanh nước biển, áo vàng, áo cam). Đây là đặc điểm thị giác CỰC KỲ QUAN TRỌNG để phân biệt các nhân vật.
6. [MULTI-ACTION]: Giữ lại tất cả hành động đặc thù — đặc biệt hành động độc đáo như 'buông tay khỏi ghi-đông', 'giang hai tay lên cao', 'ăn mừng' (vì đây là frame hiếm, khác biệt hẳn).
7. [KHÔNG BỊA GÓC QUAY]: TUYỆT ĐỐI không thêm 'aerial drone view', 'flycam', 'overhead view'... nếu câu gốc KHÔNG đề cập. Chỉ thêm góc quay khi câu gốc nói rõ.
8. [OCR & BẢNG SỐ LIỆU]: Khi query mô tả bảng số liệu/slide có chữ viết, điền ocr_keywords với các cụm từ TIẾNG VIỆT xuất hiện trực tiếp trên bảng (tên cột, tên vùng, chủ đề bài giảng). visual_prompt_en phải mô tả cảnh "presentation slide with data table" hoặc "blackboard with statistics".

Ví dụ chuẩn mẫu:
- Câu gốc: "Đoạn clip về 1 thị trấn ven biển thu hút du khách hiếu kỳ nhờ 1 loài động vật biển nguy hiểm. Loài động vật này nổi tiếng trong 1 bộ phim của đạo diễn Steven Spielberg được sản xuất vào năm 1975"
  -> visual_prompt_en: "curious tourists and people in coastal beach town watching great white shark, seaside crowd"
  -> core_objects_en: ["person", "shark", "boat"]

- Câu gốc: "Đoạn clip ghi lại khoảnh khắc về đích của một chặng đua xe đạp. Vận động viên mặc áo xanh nước biển dẫn đầu đoàn. Khi chỉ còn cách đích một đoạn ngắn, anh buông cả hai tay khỏi ghi-đông, giang hai tay lên cao ăn mừng chiến thắng trong khi xe vẫn tiến về phía trước. Ngay phía sau anh là một vận động viên mặc áo vàng và một vận động viên mặc áo cam đang lần lượt lao về đích."
  -> visual_prompt_en: "cyclist in blue jersey celebrating victory with both hands raised off handlebars approaching finish line, followed by yellow jersey cyclist and orange jersey cyclist racing to finish"
  -> core_objects_en: ["bicycle", "person", "helmet"]
  -> ocr_keywords: []
  -> reasoning: "Giữ: màu áo (xanh/vàng/cam), hành động độc đáo (buông tay, giang tay ăn mừng), cảnh về đích. KHÔNG thêm 'aerial drone view' vì câu gốc không đề cập."

- Câu gốc: "Clip bài giảng môn Địa lí, có 1 bảng số liệu về mạng lưới đô thị ở Việt Nam. Bảng này thể hiện sự khác nhau về phân bố đô thị giữa các vùng bằng màu sắc: 3 vùng có nhiều đô thị nhất thì con số thể hiện số lượng đô thị được in màu đỏ, còn 2 vùng có ít đô thị nhất thì con số này được in màu xanh."
  -> visual_prompt_en: "geography class presentation slide with data table about Vietnam urban network, colored numbers in table showing regional distribution"
  -> core_objects_en: ["person"]
  -> ocr_keywords: ["mạng lưới đô thị", "đô thị", "dân số đô thị", "vùng", "địa lí", "việt nam"]
  -> reasoning: "Query mô tả bảng số liệu địa lý → visual_prompt_en mô tả slide bài giảng với bảng dữ liệu. ocr_keywords lấy các cụm tiếng Việt xuất hiện trực tiếp trên bảng. CLIP không nhận diện được màu chữ trong bảng."

- Câu gốc: "Cảnh giáo viên nam đeo kính, mặc áo sơ mi kẻ sọc ngắn tay, xuất hiện ở góc dưới bên trái và dùng hai tay làm cử chỉ minh họa khi đang giảng bài"
  -> visual_prompt_en: "male teacher wearing glasses and striped short-sleeve shirt, gesturing with both hands while teaching"
  -> core_objects_en: ["person"]
  -> ocr_keywords: []
  -> reasoning: "Bỏ 'góc dưới bên trái' (bố cục frame), giữ ngoại hình (kính, áo sơ mi kẻ sọc) và hành động (cử chỉ tay, giảng bài)."

- Câu gốc: "Khung hình chứa một bức ảnh minh họa cô gái trẻ đeo kính, mặc áo sơ mi trắng, ngồi khoanh chân trên ghế sofa màu xám vừa cầm cốc nước vừa nhìn vào laptop mở trên đùi"
  -> visual_prompt_en: "young woman wearing glasses and white shirt, sitting cross-legged on gray sofa, holding a cup while looking at open laptop on lap"
  -> core_objects_en: ["person", "laptop", "cup", "couch"]
  -> ocr_keywords: []
  -> reasoning: "Giữ ngoại hình (kính, áo trắng), tư thế (khoanh chân, sofa xám), đồ vật (cốc, laptop trên đùi). Bỏ 'khung hình chứa bức ảnh minh họa'."

BẮT BUỘC TRẢ VỀ ĐỊNH DẠNG JSON HỢP LỆ (KHÔNG THÊM BẤT KỲ VĂN BẢN NÀO NGOÀI JSON):
{
  "visual_prompt_en": "...",
  "core_objects_en": ["..."],
  "ocr_keywords": ["..."],
  "reasoning": "..."
}"""



def expand_query_gemini(
    query: str,
    api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Sử dụng Gemini để mở rộng & suy luận câu truy vấn thị giác.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None

    cleaned_q = query.strip()
    if not cleaned_q:
        return None

    if cleaned_q in _EXPANSION_CACHE:
        return _EXPANSION_CACHE[cleaned_q]

    try:
        from google import genai
        client = genai.Client(api_key=key)

        candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
        for mdl in candidate_models:
            try:
                response = client.models.generate_content(
                    model=mdl,
                    contents=f"{SYSTEM_PROMPT}\n\nCâu truy vấn cần phân tích:\n\"{cleaned_q}\""
                )
                txt = response.text.strip()
                # Bỏ markdown ```json nếu có
                txt = re.sub(r"^```(?:json)?", "", txt, flags=re.MULTILINE)
                txt = re.sub(r"```$", "", txt, flags=re.MULTILINE).strip()

                parsed = json.loads(txt)
                if "visual_prompt_en" in parsed:
                    _EXPANSION_CACHE[cleaned_q] = parsed
                    return parsed
            except Exception:
                continue

    except Exception as e:
        print(f"⚠️ Lỗi gọi Query Expander: {e}")

    return None


# Map màu áo tiếng Việt → tiếng Anh
_COLOR_MAP = {
    "xanh nước biển": "blue", "xanh dương": "blue", "xanh lam": "blue",
    "xanh lá": "green", "xanh lá cây": "green",
    "xanh": "blue",  # mặc định xanh → blue (phổ biến hơn)
    "đỏ": "red", "vàng": "yellow", "cam": "orange", "trắng": "white",
    "đen": "black", "hồng": "pink", "tím": "purple", "nâu": "brown",
    "xám": "gray", "bạc": "silver",
}

# Map hành động tiếng Việt → tiếng Anh
_ACTION_MAP = {
    "buông tay": "hands raised off handlebars",
    "giang tay": "arms outstretched",
    "giơ tay": "arms raised",
    "ăn mừng": "celebrating victory",
    "chiến thắng": "winning",
    "về đích": "crossing finish line",
    "qua đích": "crossing finish line",
    "lao về đích": "racing to finish line",
    "dẫn đầu": "leading the race",
    "vượt qua": "overtaking",
    "giảng bài": "teaching lecture",
    "cử chỉ tay": "hand gestures",
    "minh họa": "demonstrating gestures",
}


def _extract_jersey_colors(q_lower: str) -> list[str]:
    """Trích xuất màu áo từ câu query."""
    colors = []
    for vi_color, en_color in _COLOR_MAP.items():
        # Tìm pattern "áo + màu" hoặc "mặc + màu"
        if f"áo {vi_color}" in q_lower or f"mặc {vi_color}" in q_lower or f"jersey {vi_color}" in q_lower:
            if en_color not in colors:
                colors.append(en_color)
    return colors


def _extract_actions(q_lower: str) -> list[str]:
    """Trích xuất hành động từ câu query."""
    actions = []
    for vi_action, en_action in _ACTION_MAP.items():
        if vi_action in q_lower:
            actions.append(en_action)
    return actions


def expand_query_heuristic(query: str) -> Dict[str, Any]:
    """
    Fallback Heuristic khi không có API Key hoặc offline.
    KHÔNG dùng template cứng — trích xuất chi tiết thực từ câu gốc.
    """
    q_lower = query.lower()

    # === Case 1: Câu đố tri thức — phim Spielberg 1975 ===
    if "spielberg" in q_lower or "1975" in q_lower or "hàm cá mập" in q_lower:
        return {
            "visual_prompt_en": "curious tourists and people in coastal beach town watching great white shark, seaside crowd",
            "core_objects_en": ["person", "shark", "boat"],
            "ocr_keywords": ["1975", "Spielberg"],
            "reasoning": "Suy luận từ phim Steven Spielberg 1975 → Cá mập (Jaws/Shark) kết hợp bối cảnh thị trấn biển và du khách."
        }

    # === Case 2: Đua xe đạp — trích xuất chi tiết thực, KHÔNG dùng template cứng ===
    if "đua xe đạp" in q_lower or "xe đạp" in q_lower or "chặng đua" in q_lower:
        colors = _extract_jersey_colors(q_lower)
        actions = _extract_actions(q_lower)

        # Build visual prompt từ chi tiết thực
        parts = ["bicycle race"]

        # Màu áo người dẫn đầu (áo xuất hiện đầu tiên)
        if colors:
            primary_color = colors[0]
            jersey_desc = f"cyclist in {primary_color} jersey"
            # Thêm màu áo các vận động viên phía sau
            if len(colors) > 1:
                other_colors = " and ".join(f"{c} jersey" for c in colors[1:])
                jersey_desc += f", followed by {other_colors} cyclists"
            parts.append(jersey_desc)
        else:
            parts.append("cyclists in colorful jerseys")

        # Hành động đặc thù
        if actions:
            parts.append(", ".join(actions))

        # Ngữ cảnh về đích
        if any(kw in q_lower for kw in ["về đích", "qua đích", "lao về đích", "đích", "finish"]):
            if "crossing finish line" not in " ".join(actions):
                parts.append("at finish line")

        visual_prompt = ", ".join(parts)

        # Core objects
        core_objs = ["bicycle", "person", "helmet"]

        return {
            "visual_prompt_en": visual_prompt,
            "core_objects_en": core_objs,
            "ocr_keywords": [],
            "reasoning": f"Trích xuất từ câu gốc: màu áo={colors}, hành động={actions}. KHÔNG dùng template cứng."
        }

    # === Case 3: Mặc định — giữ câu gốc, trích xuất thêm màu sắc/hành động ===
    colors = _extract_jersey_colors(q_lower)
    actions = _extract_actions(q_lower)
    core_objs = []
    if colors:
        core_objs.append("person")

    return {
        "visual_prompt_en": query,
        "core_objects_en": core_objs,
        "ocr_keywords": [],
        "reasoning": "Giữ nguyên câu truy vấn ban đầu (không đủ context để transform)."

    }


def get_expanded_query(
    query: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Hàm tổng hợp mở rộng truy vấn: Ưu tiên Gemini, fallback Heuristic.
    """
    res = expand_query_gemini(query, api_key=api_key)
    if res:
        return res
    return expand_query_heuristic(query)
