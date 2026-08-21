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

NGUYÊN TẮC GIỮ TRỌN VẸN 3 TRỤ CỘT THỊ GIÁC (RẤT QUAN TRỌNG):
1. [CHỦ THỂ ĐÃ GIẢI MÃ]: Giải mã tri thức / đố mẹo (vd: 'phim của Steven Spielberg năm 1975' -> 'great white shark', 'vận động viên áo xanh trắng' -> 'cyclist wearing blue and white jersey').
2. [BỐI CẢNH / MÔI TRƯỜNG]: BẮT BUỘC giữ lại bối cảnh không gian (vd: 'thị trấn ven biển' -> 'coastal beach town / seaside', 'góc quay flycam trên cao' -> 'aerial drone view').
3. [CON NGƯỜI & HÀNH ĐỘNG]: BẮT BUỘC giữ lại sự hiện diện của con người/du khách và tương tác (vd: 'du khách hiếu kỳ' -> 'curious tourists / crowd of people visiting', 'đang vượt ba xe khác' -> 'overtaking competitors').

Ví dụ chuẩn mẫu:
- Câu gốc: "Đoạn clip về 1 thị trấn ven biển thu hút du khách hiếu kỳ nhờ 1 loài động vật biển nguy hiểm. Loài động vật này nổi tiếng trong 1 bộ phim của đạo diễn Steven Spielberg được sản xuất vào năm 1975"
  -> visual_prompt_en: "curious tourists and people in coastal beach town watching great white shark, seaside crowd"
  -> core_objects_en: ["person", "shark", "boat"]

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


def expand_query_heuristic(query: str) -> Dict[str, Any]:
    """
    Fallback Heuristic khi không có API Key hoặc offline.
    """
    q_lower = query.lower()
    
    # Kiểm tra một số case phổ biến
    if "spielberg" in q_lower or "1975" in q_lower or "hàm cá mập" in q_lower:
        return {
            "visual_prompt_en": "curious tourists and people in coastal beach town watching great white shark, seaside crowd",
            "core_objects_en": ["person", "shark", "boat"],
            "ocr_keywords": ["1975", "Spielberg"],
            "reasoning": "Suy luận từ phim Steven Spielberg 1975 -> Cá mập (Jaws/Shark) kết hợp bối cảnh thị trấn biển và du khách."
        }
        
    if "đua xe đạp" in q_lower or "xe đạp" in q_lower:
        return {
            "visual_prompt_en": "aerial drone view of bicycle race, cyclist in blue and white jersey overtaking competitors",
            "core_objects_en": ["bicycle", "person", "helmet"],
            "ocr_keywords": [],
            "reasoning": "Trích xuất góc quay flycam trên cao, cuộc đua xe đạp và áo xanh trắng."
        }

    return {
        "visual_prompt_en": query,
        "core_objects_en": [],
        "ocr_keywords": [],
        "reasoning": "Giữ nguyên câu truy vấn ban đầu."
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
