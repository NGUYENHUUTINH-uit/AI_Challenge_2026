import json

def calculate_recall_at_k(retrieved_items, ground_truth_items, k):
    """
    Tính chỉ số Recall@K.
    Trong AIC, R@K là 1 nếu item đúng nằm trong top K kết quả trả về, ngược lại là 0.
    Vì có thể có nhiều ground truth hợp lệ cho 1 câu hỏi, nếu TÌM THẤY ÍT NHẤT 1 ground truth trong top K => 1.
    
    Args:
        retrieved_items (list): Danh sách các kết quả (video_id hoặc frame_id) trả về, xếp theo điểm giảm dần.
        ground_truth_items (set): Tập hợp các kết quả đúng.
        k (int): Ngưỡng K.
        
    Returns:
        int: 1 hoặc 0
    """
    top_k_results = retrieved_items[:k]
    # Kiểm tra xem có bất kỳ kết quả nào trong top K nằm trong ground truth không
    for item in top_k_results:
        if item in ground_truth_items:
            return 1
    return 0

def evaluate_retrieval(predictions, ground_truths, k_values=[1, 5, 20, 50, 100]):
    """
    Đánh giá R@K và Final Score cho toàn bộ batch.
    
    Args:
        predictions (dict): { "query_id_1": ["vid1", "vid2", ...], ... }
        ground_truths (dict): { "query_id_1": ["vid1", "vid5", ...], ... }
        k_values (list): Danh sách các K để đánh giá.
        
    Returns:
        dict: Kết quả đánh giá
    """
    results = {f"R@{k}": 0.0 for k in k_values}
    num_queries = len(ground_truths)
    
    if num_queries == 0:
        return results
        
    for query_id, gt_items in ground_truths.items():
        gt_set = set(gt_items)
        pred_items = predictions.get(query_id, [])
        
        for k in k_values:
            r_at_k = calculate_recall_at_k(pred_items, gt_set, k)
            results[f"R@{k}"] += r_at_k
            
    # Trung bình trên tổng số câu truy vấn
    for k in k_values:
        results[f"R@{k}"] = results[f"R@{k}"] / num_queries
        
    # Final Score: Trung bình của R@1, R@5, R@20, R@50, R@100
    final_score = sum(results[f"R@{k}"] for k in k_values) / len(k_values)
    results["Final Score"] = final_score
    
    return results

if __name__ == "__main__":
    # Test script giả lập
    dummy_ground_truths = {
        "q1": ["video_001"],
        "q2": ["video_045", "video_046"], # Có thể đáp án nằm rải rác
        "q3": ["video_099"]
    }
    
    dummy_predictions = {
        "q1": ["video_002", "video_003", "video_001", "video_004"], # Đúng ở top 3 => R@1=0, R@5=1
        "q2": ["video_045", "video_010", "video_011"], # Đúng ở top 1 => R@1=1
        "q3": ["video_001", "video_002", "video_003"] # Không có trong kết quả => R=0
    }
    
    print("--- AIC 2026 Evaluation Test ---")
    metrics = evaluate_retrieval(dummy_predictions, dummy_ground_truths)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
