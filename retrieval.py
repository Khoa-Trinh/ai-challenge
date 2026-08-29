import os
from model import encode_text_queries


def retrieve_kis(
    processor,
    model,
    index,
    manifest,
    global_map,
    metadata,
    video_ids,
    bm25_model,
    query_en_list,
    metadata_keywords="",
    top_k=100,
    spread_window=2,
    vector_search_top_k=500,
    bm25_boost_weight=0.20,
    device="cuda",
):
    """
    - query_en_list: Danh sách các sub-query tiếng Anh mô tả chi tiết các phân cảnh.
    - metadata_keywords: Từ khóa tiếng Việt/Anh để search trong media-info (nếu có).
    - spread_window: Lấy thêm các frame xung quanh frame khớp nhất.
    """
    # 1. Encode danh sách Text Sub-queries
    query_vec = encode_text_queries(processor, model, query_en_list, device=device)

    # 2. Vector Search (Lấy top candidates)
    scores, indices = index.search(query_vec, vector_search_top_k)
    scores, indices = scores[0], indices[0]

    # 3. Metadata Boosting qua BM25
    bm25_bonus = {}
    if metadata_keywords.strip():
        tokenized_kw = metadata_keywords.lower().split()
        bm25_scores = bm25_model.get_scores(tokenized_kw)
        max_b = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        for vid, b_score in zip(video_ids, bm25_scores):
            if b_score > 0:
                bm25_bonus[vid] = (b_score / max_b) * bm25_boost_weight  # Boost tối đa bm25_boost_weight điểm

    # 4. Gộp điểm và gom nhóm
    ranked_candidates = []
    seen_keys = set()

    for score, idx in zip(scores, indices):
        kf_key = manifest[idx]  # Ví dụ: "L21_V001/005.jpg"
        v_id, kf_name = kf_key.split("/")

        final_score = float(score) + bm25_bonus.get(v_id, 0.0)
        map_info = global_map.get(kf_key, {})
        frame_idx = map_info.get("frame_idx", 0) if isinstance(map_info, dict) else map_info

        if kf_key not in seen_keys:
            seen_keys.add(kf_key)
            ranked_candidates.append({
                "video": v_id,
                "kf_name": kf_name,
                "kf_int": int(kf_name.replace(".jpg", "")),
                "frame_idx": frame_idx,
                "score": final_score,
            })

    # 5. Sắp xếp lại theo điểm
    ranked_candidates.sort(key=lambda x: x["score"], reverse=True)

    # 6. Window Spreading (Gắp thêm các frame xung quanh top video để phủ trọn Ground Truth)
    submission_rows = []
    submitted_pairs = set()

    for cand in ranked_candidates:
        v_id = cand["video"]
        base_kf = cand["kf_int"]

        # Thêm chính nó và các frame xung quanh
        for offset in range(-spread_window, spread_window + 1):
            target_kf_int = base_kf + offset
            target_key = f"{v_id}/{target_kf_int:03d}.jpg"

            if target_key in global_map:
                m_info = global_map[target_key]
                f_idx = m_info.get("frame_idx", 0) if isinstance(m_info, dict) else m_info
                pair = (v_id, f_idx)

                if pair not in submitted_pairs:
                    submitted_pairs.add(pair)
                    submission_rows.append({
                        "video": v_id,
                        "frame_idx": f_idx,
                        "score": cand["score"] - abs(offset) * 0.001,
                    })

            if len(submission_rows) >= top_k:
                break
        if len(submission_rows) >= top_k:
            break

    return submission_rows[:top_k]


def save_csv(results, query_id, output_dir="/kaggle/working/submissions"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{query_id}.csv")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"{r['video']}, {r['frame_idx']}\n")
    print(f"-> Đã xuất: {out_path} ({len(results)} dòng)")
    return out_path
