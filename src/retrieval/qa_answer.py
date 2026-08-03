"""
qa_answer.py — Xử lý dạng truy vấn Q&A (Visual Question Answering).

BÀI TOÁN MỚI (từ đề thi chính thức AIC 2026): Q&A yêu cầu không chỉ định vị
đúng khoảnh khắc, mà còn phải TRẢ LỜI câu hỏi về nội dung trong khoảnh khắc
đó (vd. "có bao nhiêu người lên sân khấu?"). Đây là bài toán Visual Question
Answering (VQA) — khác hoàn toàn với Retrieval (tìm kiếm) mà B4/B5 đã làm.

CÁCH TIẾP CẬN: dùng khả năng đọc ảnh (vision) của Claude — gửi kèm keyframe
đại diện (ảnh) + câu hỏi, để model tự mô tả và trả lời trực tiếp. Không cần
train riêng 1 mô hình VQA — đây là cách tận dụng LLM đa phương thức có sẵn,
phù hợp ràng buộc thời gian/ngân sách của nhóm (giống lý do chọn CQR thay vì
train CDR đã phân tích trước đây).
"""
import base64
from pathlib import Path
from typing import Callable, Optional


def _encode_image(image_path: str) -> tuple[str, str]:
    """Đọc file ảnh, trả về (base64_data, media_type)."""
    path = Path(image_path)
    ext = path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(ext, "image/jpeg")

    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def make_answer_fn(client, model: str = "claude-sonnet-5") -> Callable[[str, str], str]:
    """
    Factory tạo answer_fn(image_path, question) -> answer (str), dùng Claude
    đọc ảnh thật. Dùng model có khả năng vision (Sonnet/Opus) — KHÔNG dùng
    Haiku cho việc này vì cần độ chính xác thị giác cao hơn, khác với
    planner.py (chỉ cần phân loại text, Haiku đã đủ).

    Trả None nếu ảnh không đọc được (vd. đường dẫn dummy chưa có file thật)
    — để agent.py xử lý fallback, không crash cả pipeline.
    """
    def answer_fn(image_path: str, question: str) -> Optional[str]:
        try:
            image_data, media_type = _encode_image(image_path)
        except (FileNotFoundError, OSError):
            return None  # fallback: chưa có ảnh thật (data dummy) hoặc lỗi đọc file

        response = client.messages.create(
            model=model,
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type, "data": image_data,
                    }},
                    {"type": "text", "text": (
                        f"Đây là 1 khung hình từ video. Trả lời câu hỏi sau NGẮN GỌN, "
                        f"chỉ dựa trên những gì thấy được trong ảnh, không suy đoán thêm:\n"
                        f"{question}"
                    )},
                ],
            }],
        )
        return response.content[0].text.strip()

    return answer_fn


def make_mock_answer_fn(fixed_answer: str = "[mock answer]") -> Callable[[str, str], str]:
    """
    answer_fn giả lập — dùng để TEST toàn bộ pipeline Q&A (bao gồm việc chọn
    đúng frame đại diện, đúng luồng gọi) mà KHÔNG cần API key/ảnh thật.
    Tương tự tinh thần retriever_stub trước đây: tách rời phần "có logic
    đúng luồng" khỏi "có kết quả đúng thật".
    """
    def answer_fn(image_path: str, question: str) -> str:
        return fixed_answer
    return answer_fn