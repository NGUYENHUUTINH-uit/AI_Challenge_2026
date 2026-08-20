"""
ai_vqa_service.py — Module dịch vụ tích hợp Google Gemini AI cho tính năng Q&A.
- Tự động nạp cấu hình bảo mật từ file .env 
- Cung cấp hàm gemini_vqa_inference() để truyền vào tham số answer_fn của SearchPipeline.
"""

import os
from pathlib import Path
from PIL import Image
from google import genai
from dotenv import load_dotenv

load_dotenv()

def gemini_vqa_inference(image_path: str, question: str) -> str:
    """
    Hàm xử lý VQA sử dụng Gemini API với cơ chế chống lỗi đường dẫn và tự động fallback.
    Đảm bảo an toàn tuyệt đối cho hệ thống khi gặp file ảnh bị thiếu trên ổ đĩa.
    """
    if not image_path:
        return "Lỗi: Không nhận được đường dẫn hình ảnh từ bộ định vị."
    
    if not question or not question.strip():
        return "Lỗi: Câu hỏi Q&A đang bị để trống."

    try:
        # 1. Chuẩn hóa chuỗi đường dẫn (xử lý triệt để dấu gạch chéo ngược Windows/Linux)
        cleaned_path = str(image_path).replace("\\", "/")
        img_file = Path(cleaned_path)
        
        # 2. Xử lý đường dẫn tương đối hoặc tuyệt đối
        if not img_file.is_absolute():
            # Dò tìm dựa trên thư mục hiện tại của hệ thống hoặc workspace
            current_working_dir = Path(os.getcwd())
            potential_file = current_working_dir / cleaned_path
            if potential_file.exists():
                img_file = potential_file

        # 3. Kiểm tra thực tế file có tồn tại vật lý trên ổ đĩa hay không
        if not img_file.exists():
            # Trả về thông báo chuyên nghiệp, giúp debug nhanh mà không làm crash app
            return f"Cảnh báo: Khung hình chưa được tải về hoặc không tồn tại tại ({img_file.name})."

        # 4. Khởi tạo Gemini Client an toàn
        client = genai.Client()
        
        # 5. Mở file ảnh bằng PIL
        with Image.open(img_file) as img:
            prompt = (
                f"Dựa vào bức ảnh được cung cấp từ video, hãy trả lời câu hỏi sau "
                f"một cách cực kỳ ngắn gọn, súc tích (ưu tiên số lượng, màu sắc hoặc hành động), "
                f"trả thẳng vào trọng tâm bằng tiếng Việt:\n"
                f"Câu hỏi: {question.strip()}\n"
                f"Trả lời:"
            )
            
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[img, prompt]
            )
            
            if response and response.text:
                return response.text.strip()
            
        return "AI không đưa ra được câu trả lời cho khung hình này."
        
    except Exception as e:
        # Bắt mọi ngoại lệ phát sinh trong quá trình gọi API hoặc xử lý ảnh
        return f"Lỗi hệ thống VQA: {type(e).__name__}"