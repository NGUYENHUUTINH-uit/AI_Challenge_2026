"""
rerank.py \u2014 Object-based re-ranking sau FAISS search
=====================================================
Nh\u1eadn k\u1ebft qu\u1ea3 th\u00f4 t\u1eeb retriever.py, boost score n\u1ebfu frame
c\u00f3 ch\u1ee9a c\u00e1c object kh\u1edbp v\u1edbi query.

Ho\u1ea1t \u0111\u1ed9ng:
  1. Tr\u00edch xu\u1ea5t t\u1eeb kh\u00f3a Vi\u1ec7t \u2192 COCO class (person, car, ...)
  2. \u0110\u1ebfm s\u1ed1 object kh\u1edbp gi\u1eefa query v\u00e0 objects c\u1ee7a frame
  3. Boost score: new_score = clip_score + match_bonus
  4. S\u1eafp x\u1ebfp l\u1ea1i theo new_score gi\u1ea3m d\u1ea7n
"""

# ============================================================
# T\u1eea \u0110I\u1ec2N VI\u1ec6T \u2192 COCO 80 CLASSES
# ============================================================
_VI_TO_COCO = {
    # Ng\u01b0\u1eddi
    "ng\u01b0\u1eddi":      "person",
    "\u0111\u00e0n \u00f4ng":   "person",
    "ph\u1ee5 n\u1eef":    "person",
    "b\u00e0 l\u00e3o":     "person",
    "\u00f4ng l\u00e3o":     "person",
    "\u0111\u1ee9a tr\u1ebb":   "person",
    "tr\u1ebb em":    "person",
    "c\u1ea3nh s\u00e1t":  "person",
    "b\u00e1c s\u0129":     "person",
    # Ph\u01b0\u01a1ng ti\u1ec7n
    "xe \u0111\u1ea1p":    "bicycle",
    "xe \u0111\u1ea1p \u0111i\u1ec7n": "bicycle",
    "xe m\u00e1y":   "motorcycle",
    "m\u00f4t\u00f4":      "motorcycle",
    "xe h\u01a1i":    "car",
    "\u00f4 t\u00f4":       "car",
    "xe \u00f4 t\u00f4":   "car",
    "xe bu\u00fdt":   "bus",
    "xe t\u1ea3i":    "truck",
    "xe l\u1eeda":    "train",
    "m\u00e1y bay":   "airplane",
    "thuy\u1ec1n":     "boat",
    # \u0110\u1ed9ng v\u1eadt
    "ch\u00f3":        "dog",
    "m\u00e8o":        "cat",
    "ng\u1ef1a":       "horse",
    "b\u00f2":         "cow",
    "c\u1eeba":        "sheep",
    "gà":         "bird",
    "chim":       "bird",
    "voi":        "elephant",
    # \u0110\u1ed3 v\u1eadt
    "\u0111i\u1ec7n tho\u1ea1i": "cell phone",
    "\u0111i\u1ec7n tho\u1ea1i th\u00f4ng minh": "cell phone",
    "laptop":     "laptop",
    "m\u00e1y t\u00ednh":  "laptop",
    "ba l\u00f4":      "backpack",
    "t\u00faui x\u00e1ch":   "handbag",
    "nh\u00f3m":       "umbrella",
    "gh\u1ebf":        "chair",
    "b\u00e0n":         "dining table",
    "s\u00e1ch":       "book",
    "chai":       "bottle",
    "c\u1ed1c":        "cup",
    "k\u00e9o":        "scissors",
    "b\u00f3ng":       "sports ball",
    "b\u00f3ng \u0111\u00e1":   "sports ball",
    "\u0111\u1ed3ng h\u1ed3":  "clock",
}


def extract_objects_from_query(query_text: str) -> list:
    """
    Tr\u00edch xu\u1ea5t COCO class names t\u1eeb c\u00e2u query ti\u1ebfng Vi\u1ec7t + ti\u1ebfng Anh.
    Tr\u1ea3 v\u1ec1 list string, v\u00ed d\u1ee5: ["person", "bicycle"]
    """
    query_lower = query_text.lower()
    objects = set()

    # T\u00ecm t\u1eeb kh\u00f3a ti\u1ebfng Vi\u1ec7t
    for vi_word, coco_cls in _VI_TO_COCO.items():
        if vi_word in query_lower:
            objects.add(coco_cls)

    # T\u00ecm tr\u1ef1c ti\u1ebfp COCO class name ti\u1ebfng Anh trong query
    coco_classes = set(_VI_TO_COCO.values())
    for cls in coco_classes:
        if cls in query_lower:
            objects.add(cls)

    return list(objects)


def rerank_by_objects(results: list, query_text: str,
                      object_boost: float = 0.05) -> list:
    """
    Re-rank k\u1ebft qu\u1ea3 FAISS d\u1ef1a tr\u00ean object match.

    Args:
        results:      list of dict t\u1eeb retriever.search_keyframes()
                      m\u1ed7i dict c\u00f3 tr\u01b0\u1eddng "objects" (list class name) v\u00e0 "score"
        query_text:   c\u00e2u query g\u1ed1c \u0111\u1ec3 tr\u00edch xu\u1ea5t object c\u1ea7n t\u00ecm
        object_boost: \u0111i\u1ec3m c\u1ed9ng th\u00eam cho m\u1ed7i object kh\u1edbp (default 0.05)

    Returns:
        list k\u1ebft qu\u1ea3 \u0111\u00e3 s\u1eafp x\u1ebfp l\u1ea1i theo score gi\u1ea3m d\u1ea7n
    """
    query_objects = extract_objects_from_query(query_text)

    if not query_objects:
        # Kh\u00f4ng t\u00ecm th\u1ea5y object n\u00e0o \u2192 gi\u1eef nguy\u00ean th\u1ee9 t\u1ef1 FAISS
        return results

    reranked = []
    for res in results:
        frame_objects = set(res.get("objects", []))
        match_count   = sum(1 for obj in query_objects if obj in frame_objects)
        bonus         = match_count * object_boost
        new_score     = res.get("score", 0.0) + bonus

        new_res = {**res, "score": round(new_score, 4)}
        reranked.append(new_res)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked


if __name__ == "__main__":
    # Test nhanh
    dummy = [
        {"video_id": "L25_V001", "filename": "0001.jpg", "score": 0.82, "objects": ["person", "car"]},
        {"video_id": "L25_V001", "filename": "0002.jpg", "score": 0.85, "objects": ["bicycle"]},
        {"video_id": "L25_V001", "filename": "0003.jpg", "score": 0.80, "objects": ["person", "bicycle"]},
    ]
    query = "ng\u01b0\u1eddi \u0111i xe \u0111\u1ea1p"
    reranked = rerank_by_objects(dummy, query)
    print("Query:", query)
    print("Objects extracted:", extract_objects_from_query(query))
    for r in reranked:
        print(f"  {r['filename']}: score={r['score']:.4f} objects={r['objects']}")

