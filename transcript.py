import re
import bisect


def clean_vietnamese_text(text):
    """
    Chuẩn hóa văn bản tiếng Việt để tìm kiếm: chữ thường, loại bỏ ký tự đặc biệt.
    """
    if not text:
        return ""
    # Chuyển chữ thường và làm sạch khoảng trắng
    cleaned = re.sub(r'[\r\n\t]+', ' ', text.lower())
    return " ".join(cleaned.split())


def search_transcripts(transcripts, query, global_map, max_results=10):
    """
    Tìm kiếm các đoạn thoại trong video khớp với từ khóa trong câu query.
    
    Args:
        transcripts (dict): Dữ liệu từ video_transcripts.json
        query (str): Câu query của người dùng (tiếng Việt hoặc tiếng Anh)
        global_map (dict): Dữ liệu ánh xạ keyframes từ global_map_keyframes.json
        max_results (int): Số lượng kết quả transcript tối đa trả về
        
    Returns:
        list of dict: Danh sách các match gồm v_id, kf_name, start_sec, text, score
    """
    if not transcripts or not query:
        return []

    q_clean = clean_vietnamese_text(query)
    q_words = [w for w in q_clean.split() if len(w) > 1]
    if not q_words:
        return []

    matches = []

    # Chuẩn bị dữ liệu timeline từ global_map để tìm nearest keyframe
    # Nếu global_map có sẵn video_timelines thì dùng trực tiếp
    timelines = {}
    if isinstance(global_map, dict) and "video_timelines" in global_map:
        timelines = global_map["video_timelines"]
    else:
        # Tự động dựng timeline từ keyframes_map nếu cần
        kf_map = global_map.get("keyframes_map", global_map) if isinstance(global_map, dict) else {}
        for kf_key, info in kf_map.items():
            if "/" in kf_key and isinstance(info, dict):
                v_id, kf_name = kf_key.split("/", 1)
                if v_id not in timelines:
                    timelines[v_id] = {"keyframes": [], "pts_times": []}
                timelines[v_id]["keyframes"].append(kf_name)
                timelines[v_id]["pts_times"].append(info.get("pts_time", 0.0))

    # Tìm kiếm qua từng video
    for v_id, v_data in transcripts.items():
        if not isinstance(v_data, dict):
            continue

        segments = v_data.get("segments", [])
        title = v_data.get("title", "")
        watch_url = v_data.get("watch_url", "")

        for seg in segments:
            seg_text = seg.get("text", "")
            seg_start = seg.get("start", 0.0)
            seg_clean = clean_vietnamese_text(seg_text)

            # 1. Exact Phrase Match (Khớp toàn bộ cụm từ) -> Điểm cao nhất
            if q_clean in seg_clean:
                score = 1.0
            else:
                # 2. Token Subset Match (Khớp nhiều từ khóa quan trọng)
                matched_words = [w for w in q_words if w in seg_clean]
                if len(matched_words) >= 2 or (len(q_words) == 1 and len(matched_words) == 1):
                    score = len(matched_words) / len(q_words) * 0.8
                else:
                    continue

            # Ánh xạ giây start sang Keyframe ID gần nhất
            nearest_kf = None
            if v_id in timelines:
                times = timelines[v_id].get("pts_times", [])
                kfs = timelines[v_id].get("keyframes", [])
                if times and kfs:
                    idx = bisect.bisect_left(times, seg_start)
                    idx = min(max(0, idx), len(times) - 1)
                    nearest_kf = kfs[idx]

            if not nearest_kf:
                # Fallback: ước tính theo 1 keyframe mỗi 3 giây nếu không có map
                approx_kf_idx = max(1, int(seg_start / 3.0))
                nearest_kf = f"{approx_kf_idx:03d}.jpg"

            matches.append({
                "v_id": v_id,
                "kf_name": nearest_kf,
                "key": f"{v_id}/{nearest_kf}",
                "start_sec": seg_start,
                "text": seg_text,
                "title": title,
                "watch_url": watch_url,
                "score": round(score, 3)
            })

    # Sắp xếp theo score giảm dần
    matches.sort(key=lambda x: x["score"], reverse=True)
    return matches[:max_results]
