"""
test_index.py - Test A3 output: query text -> search FAISS -> show results
Run: python src/indexing/test_index.py
"""

import sys
import json
import numpy as np
import faiss
import torch
from pathlib import Path

# Ngưỡng tối thiểu để coi là "có kết quả"
# CLIP cosine similarity: > 0.28 = khớp tốt, < 0.22 = gần như không liên quan
MIN_SCORE = 0.25

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
KEYFRAME_DIR  = PROJECT_ROOT / "data" / "processed" / "keyframes"
OUTPUT_FAISS  = PROJECT_ROOT / "data" / "processed" / "index.faiss"
OUTPUT_MAP    = PROJECT_ROOT / "data" / "processed" / "frame_index.json"
OUTPUT_FEAT   = PROJECT_ROOT / "data" / "processed" / "clip_features.npy"


# ============================================================
# 1. LOAD INDEX
# ============================================================
def load_everything():
    print("[LOAD] Loading FAISS index and mappings...")
    index = faiss.read_index(str(OUTPUT_FAISS))
    with open(OUTPUT_MAP, "r", encoding="utf-8") as f:
        frame_map = json.load(f)   # {"0": "video0_frame_0.jpg", ...}
    print(f"[LOAD] OK - {index.ntotal} vectors, dim={index.d}")
    return index, frame_map


# ============================================================
# 2. ENCODE TEXT QUERY -> VECTOR
# ============================================================
def encode_text(query: str, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        import clip
        model, _ = clip.load("ViT-B/32", device=device)
        tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            feat = model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)   # normalize L2
        return feat.cpu().numpy().astype(np.float32)        # shape: (1, 512)

    except ImportError:
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feat = model.get_text_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype(np.float32)


# ============================================================
# 3. SEARCH TOP-K
# ============================================================
def search(query: str, index, frame_map, top_k: int = 5, min_score: float = MIN_SCORE):
    print(f'\n[QUERY] "{query}"')
    print("-" * 50)

    # Encode query
    query_vec = encode_text(query)

    # Search FAISS - luon tra ve top_k ket qua
    scores, indices = index.search(query_vec, top_k)

    # Loc theo nguong MIN_SCORE
    passed = [(s, i) for s, i in zip(scores[0], indices[0]) if s >= min_score]

    if not passed:
        # Khong co ket qua nao du nguong - hien thi ro rang
        best_score = scores[0][0]
        print(f"  [NO MATCH] Best score = {best_score:.4f} < threshold {min_score}")
        print(f"  -> Khong co frame nao khop voi query nay.")
        return []

    print(f"{'Rank':<5} {'Score':>7}  Frame  |  video_id  |  timestamp")
    print("-" * 50)
    results = []
    for rank, (score, idx) in enumerate(passed, start=1):
        entry = frame_map.get(str(idx), {})
        # Tuong thich ca format moi (dict) va cu (string)
        if isinstance(entry, dict):
            frame_name = entry.get("filename", f"unknown_{idx}")
            video_id   = entry.get("video_id", "?")
            timestamp  = entry.get("timestamp", "?")
        else:
            frame_name = entry
            video_id   = "?"
            timestamp  = "?"
        print(f"  #{rank}   {score:>6.4f}  {frame_name}  |  {video_id}  |  {timestamp}s")
        results.append({"rank": rank, "score": float(score),
                        "frame": frame_name, "video_id": video_id, "timestamp": timestamp})

    return results


# ============================================================
# 4. MAIN - CHAY TEST
# ============================================================
if __name__ == "__main__":
    # Kiem tra file da ton tai chua
    for f in [OUTPUT_FAISS, OUTPUT_MAP]:
        if not f.exists():
            print(f"[ERROR] File not found: {f}")
            print("        Please run indexer.py first!")
            sys.exit(1)

    # Load index
    index, frame_map = load_everything()

    print("\n" + "=" * 50)
    print("  A3 TEST - Text-to-Frame Search")
    print("=" * 50)

    # --------------------------------------------------------
    # TEST CASES: thay doi cac cau query o day
    # --------------------------------------------------------
    test_queries = [
        "a girl holding a microphone",
        "a woman singing with a microphone",
        "female reporter holding microphone",
        "a girl speaking into a mic on stage",
    ]

    for query in test_queries:
        search(query, index, frame_map, top_k=3)

    # --------------------------------------------------------
    # INTERACTIVE MODE: tu nhap query
    # --------------------------------------------------------
    print("\n" + "=" * 50)
    print("  INTERACTIVE MODE (type 'q' to quit)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nEnter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if user_input.lower() in ("q", "quit", "exit", ""):
            print("Bye!")
            break

        search(user_input, index, frame_map, top_k=5)
