from fastapi.testclient import TestClient
import pytest
import sys
import os

# Đảm bảo python tìm thấy thư mục src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backend.api import app

client = TestClient(app)

def test_read_root():
    """Kiểm tra server có phản hồi khi gọi root endpoint không"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_search_valid_text():
    """Kiểm tra trường hợp gửi query hợp lệ"""
    payload = {
        "text": "một người đang đi chợ kính thông minh",
        "history": []
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "segments" in data
    assert "results" in data
    assert isinstance(data["segments"], list)
    assert isinstance(data["results"], list)

def test_search_empty_text():
    """Kiểm tra hệ thống có bắt lỗi query rỗng hay không (Validation cực quan trọng)"""
    payload = {
        "text": "   ",
        "history": []
    }
    response = client.post("/search", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Query text cannot be empty"

def test_search_image_invalid_format():
    """Kiểm tra hệ thống có từ chối file sai định dạng (ví dụ file .txt) không"""
    file_data = {"file": ("test.txt", b"fake text data", "text/plain")}
    response = client.post("/search-image", files=file_data)
    assert response.status_code == 400
    assert "Only PNG" in response.json()["detail"]