def expand_query(raw_query):
    """
    Sử dụng Prompt Engineering (giả lập LLM) để tự động sinh 3-5 biến thể truy vấn
    chi tiết từ câu hỏi thô của người dùng.
    
    Args:
        raw_query (str): Câu truy vấn thô ban đầu.
        
    Returns:
        list: Danh sách các biến thể truy vấn được sinh ra.
    """
    print(f"[LLM] Đang phân tích truy vấn thô: '{raw_query}'...")
    
    # Prompt giả lập (Trong thực tế sẽ gởi prompt này cho LLM API)
    prompt = f"""
    You are an AI assistant helping to search for videos.
    The user provided this raw query: "{raw_query}"
    Please generate 3-5 detailed variants of this query to improve search coverage.
    Focus on synonyms, expanded context, and descriptive terms.
    """
    
    # Dữ liệu giả lập kết quả trả về từ LLM
    raw_query_lower = raw_query.lower()
    variants = []
    
    if "xe đạp" in raw_query_lower or "bicycle" in raw_query_lower:
        variants = [
            raw_query,
            "Một người đang đạp xe trên đường",
            "Phương tiện xe thô sơ 2 bánh đang di chuyển",
            "Vận động viên xe đạp đang tập luyện"
        ]
    elif "hành động" in raw_query_lower or "action" in raw_query_lower:
        variants = [
            raw_query,
            "Cảnh rượt đuổi tốc độ cao",
            "Phim hành động võ thuật",
            "Vụ nổ lớn trong phim hành động",
            "Cảnh sát vây bắt tội phạm"
        ]
    else:
        # Generic fallback
        variants = [
            raw_query,
            f"{raw_query} (góc quay rộng)",
            f"{raw_query} (cận cảnh)",
            f"{raw_query} ban ngày",
            f"{raw_query} độ phân giải cao"
        ]
        
    print(f"[LLM] Đã sinh ra {len(variants)} biến thể:")
    for i, v in enumerate(variants, 1):
        print(f"  {i}. {v}")
        
    return variants

if __name__ == "__main__":
    test_query = "Người đàn ông mặc áo đỏ đang chạy xe đạp"
    results = expand_query(test_query)
