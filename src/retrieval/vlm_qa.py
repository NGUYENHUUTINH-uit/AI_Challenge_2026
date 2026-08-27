"""
vlm_qa.py — Vision-Language Model cho bài toán Visual Question Answering (Q&A)
=============================================================================
Hỗ trợ:
1. Google Gemini Vision (gemini-2.5-flash / gemini-1.5-flash) qua SDK `google.genai`.
2. Tối ưu prompt theo chuẩn Thể lệ cuộc thi AIC 2026:
   - Trả lời ngắn gọn, chính xác ngữ nghĩa, tối đa 100 ký tự.
   - Trả về trực tiếp con số, màu sắc, tên đối tượng, hoặc sự kiện.
3. Cơ chế Fallback Heuristic (Object Detection + OCR) khi không có API Key hoặc offline.
4. Cache kết quả tránh gọi API lặp lại.
"""

import os
import re
from pathlib import Path
from typing import Optional, Dict
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Cache lưu trữ câu trả lời: {(image_path_str, question): answer}
_QA_CACHE: Dict[tuple, str] = {}


def clean_vqa_answer(raw_answer: str) -> str:
    """Chuẩn hóa câu trả lời theo thể lệ AIC (tối đa 100 ký tự, bỏ định dạng markdown thừa)."""
    if not raw_answer:
        return ""
    
    # Bỏ markdown như **bold**, *italic*, ```
    cleaned = re.sub(r"[\*\_`]", "", raw_answer).strip()
    
    # Bỏ các tiền tố thừa thường gặp từ AI (vd: "Đáp án: ", "Trả lời: ", "Answer: ")
    cleaned = re.sub(r"^(?:Đáp án|Câu trả lời|Trả lời|Answer|Result)[:\s\-]+", "", cleaned, flags=re.IGNORECASE).strip()
    
    # Bỏ dấu chấm ở cuối nếu là câu trả lời ngắn (từ đơn)
    if len(cleaned.split()) <= 4 and cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()
        
    # Cắt ngắn nếu vượt quá 100 ký tự (theo thể lệ)
    if len(cleaned) > 100:
        cleaned = cleaned[:97] + "..."
        
    return cleaned


def answer_question_gemini(
    image_path: str | Path,
    question: str,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Optional[str]:
    """
    Sử dụng Gemini Vision để trả lời câu hỏi dựa trên ảnh keyframe.
    Tự động thử lần lượt các model siêu nhanh: gemini-2.5-flash -> gemini-2.0-flash -> gemini-1.5-flash.
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return None

    img_path = Path(image_path)
    if not img_path.exists():
        return None

    cache_key = (str(img_path.resolve()), question.strip().lower())
    if cache_key in _QA_CACHE:
        return _QA_CACHE[cache_key]

    try:
        from google import genai
        client = genai.Client(api_key=key)

        pil_img = Image.open(img_path).convert("RGB")

        prompt = f"""Bạn là trợ lý AI thi đấu cuộc thi AI Challenge (AIC 2026) chuyên trả lời câu hỏi về hình ảnh (Visual Question Answering).
Nhiệm vụ: Nhìn vào bức ảnh và trả lời câu hỏi sau bằng Tiếng Việt hoặc con số cụ thể.

Câu hỏi: "{question}"

YÊU CẦU BẮT BUỘC:
1. Trả lời cực kỳ ngắn gọn, súc tích, đi thẳng vào đáp án (dưới 10 từ, tối đa 100 ký tự).
2. Nếu hỏi số lượng: CHỈ trả về con số (ví dụ: "3" hoặc "3 người").
3. Nếu hỏi màu sắc: CHỈ trả về tên màu (ví dụ: "Màu đỏ").
4. Nếu hỏi chữ/biển số: CHỈ trả về chính xác nội dung chữ viết.
5. KHÔNG giải thích dài dòng, KHÔNG chào hỏi, KHÔNG dùng tiền tố như 'Đáp án là'."""

        # Danh sách các model đang hoạt động chính thức trên Google AI Studio
        candidate_models = [model_name] if model_name else []
        candidate_models += ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

        for mdl in candidate_models:
            if not mdl:
                continue
            try:
                response = client.models.generate_content(
                    model=mdl,
                    contents=[pil_img, prompt]
                )
                ans = clean_vqa_answer(response.text)
                if ans:
                    _QA_CACHE[cache_key] = ans
                    return ans
            except Exception as model_err:
                # Thử model tiếp theo nếu model này không tồn tại / quá tải
                continue

    except Exception as e:
        print(f"⚠️ Lỗi khởi tạo Gemini VLM: {e}")

    return None


def answer_question_heuristic(
    question: str,
    objects: Optional[list] = None,
    ocr_text: Optional[str] = None
) -> str:
    """
    Heuristic Fallback khi không có API key hoặc offline:
    Sử dụng Object Detection list và OCR text để suy đoán câu trả lời.
    """
    q_lower = question.lower()
    objs = objects or []
    ocr = ocr_text or ""

    # 1. Câu hỏi đếm số lượng
    if any(kw in q_lower for kw in ["bao nhiêu", "mấy", "số lượng", "đếm", "count"]):
        for target, synonyms in [
            ("người", ["person", "man", "woman"]),
            ("xe", ["car", "motorcycle", "bus", "truck", "bicycle"]),
            ("ô tô", ["car"]),
            ("xe máy", ["motorcycle"]),
            ("chó", ["dog"]),
            ("mèo", ["cat"]),
            ("cây", ["tree"]),
            ("ghế", ["chair", "couch", "bench"]),
        ]:
            if target in q_lower:
                cnt = sum(1 for o in objs if any(s in str(o).lower() for s in synonyms))
                if cnt > 0:
                    return str(cnt)
        if objs:
            return str(len(objs))
        return "1"

    # 2. Câu hỏi về chữ viết, biển hiệu, số
    if any(kw in q_lower for kw in ["chữ", "biển", "số", "ghi gì", "viết gì", "text"]):
        if ocr:
            return ocr[:60]

    # 3. Câu hỏi về màu sắc
    if any(kw in q_lower for kw in ["màu", "màu gì", "color"]):
        for color in ["đỏ", "xanh dương", "xanh lá", "vàng", "trắng", "đen", "cam", "hồng", "tím"]:
            if color in q_lower:
                return f"Màu {color}"

    return ""


def get_vqa_answer(
    image_path: Optional[str | Path],
    question: str,
    objects: Optional[list] = None,
    ocr_text: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Hàm tổng hợp: Ưu tiên Gemini Vision, fallback heuristic nếu không có key / lỗi.
    """
    # 1. Thử gọi Gemini Vision nếu có ảnh và API Key
    if image_path and (api_key or os.getenv("GEMINI_API_KEY")):
        gemini_ans = answer_question_gemini(image_path, question, api_key=api_key)
        if gemini_ans:
            return gemini_ans

    # 2. Fallback Heuristic
    return answer_question_heuristic(question, objects=objects, ocr_text=ocr_text)
