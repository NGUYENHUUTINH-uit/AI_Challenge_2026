"""
gemini_reranker.py — Gemini Vision Re-ranker cho AIC 2026
==========================================================
Stage 2: Sau khi CLIP tìm top-K candidates, dùng Gemini Vision
để xem thực tế từng ảnh và cho điểm chính xác hơn.

Tại sao cần:
- CLIP chỉ hiểu ngữ nghĩa tổng quát (embedding text-image similarity)
- CLIP không đếm được số lượng ("4 con cá", "3 tay đua")
- CLIP không đọc được text trên ảnh ("đường Hồ Tùng Mậu", "13 giây")
- Gemini Vision nhìn thật vào ảnh → verify chính xác từng chi tiết

Usage:
    from src.retrieval.gemini_reranker import gemini_rerank
    top_frames = await gemini_rerank(clip_results[:20], query, api_key, top_k=10)

Quota ước tính:
    Gemini 2.0 Flash: 15 RPM, 1M TPM (free tier)
    10 frames × ~300 tokens = 3000 tokens/query → ~333 queries/day free
"""

import base64
import asyncio
import os
from pathlib import Path
from typing import Optional

# Prompt template để score từng frame
_SCORE_PROMPT = """Look at this image and rate how well it matches this search description.

Search description: "{query}"

Scoring criteria:
- 1.0 = Perfect match: image clearly shows exactly what is described
- 0.7 = Good match: image mostly matches but missing some details  
- 0.4 = Partial match: image is related but doesn't clearly show the described content
- 0.1 = Poor match: image is from same general topic but doesn't match
- 0.0 = No match: completely unrelated

Important details to check:
- Count of objects (if query mentions specific numbers like "4 fish", "3 cyclists")
- Text/signs visible in the image (street names, numbers on screen, labels)
- Colors of clothing, objects
- Specific actions being performed
- Camera angle/perspective

Reply with ONLY a decimal number between 0.0 and 1.0. Nothing else."""


def _encode_image(image_path: str) -> Optional[str]:
    """Encode ảnh sang base64 để gửi lên Gemini."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _get_mime_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(suffix, "image/jpeg")


def gemini_score_frame(
    image_path: str,
    query: str,
    api_key: str,
    client=None,
) -> float:
    """
    Dùng Gemini Vision để score 1 frame theo query.
    Trả về float 0.0 - 1.0.
    """
    img_b64 = _encode_image(image_path)
    if not img_b64:
        return 0.0

    try:
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)

        prompt = _SCORE_PROMPT.format(query=query)
        # Thử model theo thứ tự ưu tiên (giảm dần chi phí/tốc độ)
        _VISION_MODELS = ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-1.5-flash"]
        response = None
        for mdl in _VISION_MODELS:
            try:
                response = client.models.generate_content(
                    model=mdl,
                    contents=[
                        {
                            "parts": [
                                {
                                    "inline_data": {
                                        "mime_type": _get_mime_type(image_path),
                                        "data": img_b64,
                                    }
                                },
                                {"text": prompt},
                            ]
                        }
                    ],
                )
                break
            except Exception as model_err:
                if "NOT_FOUND" in str(model_err) or "deprecated" in str(model_err).lower():
                    continue
                raise

        if response is None:
            return 0.0

        text = response.text.strip()
        # Parse số từ response
        import re
        match = re.search(r"\d+\.?\d*", text)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))
    except Exception as e:
        print(f"[GeminiReranker] Error scoring {image_path}: {e}")

    return 0.0


def gemini_rerank(
    frames: list[dict],
    query: str,
    api_key: Optional[str] = None,
    top_k: int = 10,
    max_verify: int = 10,
    clip_weight: float = 0.35,
    gemini_weight: float = 0.65,
) -> list[dict]:
    """
    Re-rank frames bằng Gemini Vision.
    
    Args:
        frames: List kết quả từ CLIP search (đã sorted theo score giảm dần)
        query: Câu truy vấn gốc (tiếng Việt OK)
        api_key: Gemini API key (hoặc đọc từ env)
        top_k: Số frame trả về sau re-rank
        max_verify: Số frame tối đa gửi cho Gemini (10 = nhanh, 20 = cân bằng)
        clip_weight: Trọng số CLIP score (0.35)
        gemini_weight: Trọng số Gemini score (0.65)
    
    Returns:
        List frames đã re-ranked với field 'gemini_score' và 'final_score'
    """
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        # Fallback: trả về kết quả gốc không thay đổi
        return frames[:top_k]

    if not frames:
        return []

    # Chỉ verify top max_verify frames để tiết kiệm quota
    to_verify = frames[:max_verify]
    rest = frames[max_verify:]

    # Import Gemini client 1 lần (tránh overhead)
    try:
        from google import genai
        client = genai.Client(api_key=key)
    except Exception as e:
        print(f"[GeminiReranker] Cannot init Gemini client: {e}")
        return frames[:top_k]

    # Score từng frame
    scored = []
    for frame in to_verify:
        img_path = frame.get("image_path", "")
        if img_path and Path(img_path).exists():
            g_score = gemini_score_frame(img_path, query, key, client=client)
        else:
            g_score = 0.0

        clip_score = float(frame.get("score", 0.0))
        # Normalize clip_score về [0,1] (thường là cosine similarity 0.0-1.0)
        final_score = clip_weight * clip_score + gemini_weight * g_score

        scored.append({
            **frame,
            "gemini_score": round(g_score, 4),
            "final_score": round(final_score, 4),
            "score": round(final_score, 4),  # Override score để pipeline sort đúng
        })

    # Frame không verify giữ nguyên score nhưng thêm field
    for frame in rest:
        scored.append({
            **frame,
            "gemini_score": -1.0,  # -1 = chưa verify
            "final_score": float(frame.get("score", 0.0)) * clip_weight,
            "score": float(frame.get("score", 0.0)) * clip_weight,
        })

    scored.sort(key=lambda x: -x["final_score"])
    return scored[:top_k]


def gemini_rerank_with_translation(
    frames: list[dict],
    query_vi: str,
    api_key: Optional[str] = None,
    top_k: int = 10,
    max_verify: int = 10,
) -> list[dict]:
    """
    Wrapper: tự động dịch query Việt sang Anh trước khi gửi Gemini.
    Gemini hiểu tiếng Việt nhưng prompt tiếng Anh cho kết quả ổn định hơn.
    """
    # Dịch query sang tiếng Anh để prompt Gemini chính xác hơn
    en_query = query_vi
    try:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if key:
            from google import genai
            client = genai.Client(api_key=key)
            _TRANS_MODELS = ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-1.5-flash"]
            for mdl in _TRANS_MODELS:
                try:
                    r = client.models.generate_content(
                        model=mdl,
                        contents=f"Translate this Vietnamese text to English for image search, be literal and specific:\n\"{query_vi}\"\nReturn ONLY the translation:"
                    )
                    en_query = r.text.strip().strip('"').strip("'")
                    break
                except Exception as te:
                    if "NOT_FOUND" in str(te):
                        continue
                    break
    except Exception:
        pass

    return gemini_rerank(
        frames, en_query,
        api_key=api_key,
        top_k=top_k,
        max_verify=max_verify,
    )
