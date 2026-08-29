import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from model import encode_text_queries


def inspect_query(
    processor,
    model,
    index,
    manifest,
    global_map,
    metadata,
    query_vi,
    query_en_list,
    top_n=6,
    base_kf_dir="/kaggle/input/datasets/nguynhuyds/aic-dataset",
    vector_search_top_k=200,
    max_length=64,
    device="cuda",
    show_html=True,
):
    """
    Tìm kiếm và hiển thị ảnh keyframe + Link YouTube timestamp theo layout 2 cột dọc.
    """
    print(f"🔎 Đang tìm kiếm cho: '{query_vi}'...")

    # 1. Encode text queries với padding và truncation đầy đủ
    query_vec = encode_text_queries(
        processor,
        model,
        query_en_list,
        device=device,
        max_length=max_length,
        truncation=True,
    )

    # 2. Vector Search (Lấy top candidates)
    scores, indices = index.search(query_vec, vector_search_top_k)
    scores, indices = scores[0], indices[0]

    # 3. Gom nhóm kết quả
    candidates = []
    seen_videos = {}

    for score, idx in zip(scores, indices):
        kf_key = manifest[idx]  # Ví dụ: "L21_V001/005.jpg"
        v_id, kf_name = kf_key.split("/")

        final_score = float(score)
        map_info = global_map.get(kf_key, {})

        if isinstance(map_info, dict):
            frame_idx = map_info.get("frame_idx", 0)
            pts_time = map_info.get("pts_time", 0.0)
        else:
            frame_idx = map_info
            pts_time = 0.0

        if seen_videos.get(v_id, 0) < 2:
            seen_videos[v_id] = seen_videos.get(v_id, 0) + 1
            candidates.append({
                "kf_key": kf_key,
                "video": v_id,
                "kf_name": kf_name,
                "frame_idx": frame_idx,
                "pts_time": pts_time,
                "score": final_score,
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = candidates[:top_n]

    if not top_candidates:
        print("Không tìm thấy kết quả phù hợp.")
        return

    # 4. Khởi tạo lưới hiển thị 2 cột dọc (N hàng x 2 cột)
    n_cols = 2
    n_rows = (len(top_candidates) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))

    # Đảm bảo axes luôn là mảng 2 chiều kể cả khi n_rows = 1
    if n_rows == 1:
        axes = np.array([axes])

    for i, cand in enumerate(top_candidates):
        r_idx = i // n_cols
        c_idx = i % n_cols
        ax = axes[r_idx, c_idx]

        v_id = cand["video"]
        kf_name = cand["kf_name"]
        frame_idx = cand["frame_idx"]
        pts_time = int(cand["pts_time"])
        score = cand["score"]

        meta = metadata.get(v_id, {})
        yt_url = meta.get("watch_url", "")
        title = meta.get("title", "No Title")
        timestamp_link = f"{yt_url}&t={pts_time}s" if yt_url else "#"

        # Tự động tìm đúng thư mục gốc (bao gồm cả Keyframes_L26_a, _b, _c, ...)
        prefix = v_id.split('_')[0]
        img_path = None
        for sub_tag in ["", "_a", "_b", "_c", "_d", "_e"]:
            candidate = os.path.join(base_kf_dir, f"Keyframes_{prefix}{sub_tag}", "keyframes", v_id, kf_name)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Image Not Found", ha='center', va='center')

        ax.set_title(f"Top {i+1}: {v_id}/{kf_name} | Score: {score:.3f}\nFrame: {frame_idx} | Time: {pts_time}s", fontsize=11, fontweight="bold")
        ax.axis("off")

        if show_html:
            try:
                from IPython.display import display, HTML
                display(HTML(f"""
                <div style="margin-bottom: 6px; padding: 8px 12px; border-left: 4px solid #007bff; background: #f8f9fa;">
                    <b>Top {i+1}:</b> <code>{v_id}</code> - Frame ID: <b>{frame_idx}</b> (Keyframe: <code>{kf_name}</code>)<br>
                    <b>Tiêu đề:</b> {title}<br>
                    <b>🔗 Xem video tại giây thứ {pts_time}s:</b> <a href="{timestamp_link}" target="_blank" style="color: #d9534f; font-weight: bold;">{timestamp_link}</a>
                </div>
                """))
            except Exception:
                pass

    # Tắt các ô trống nếu số lượng kết quả là số lẻ
    for j in range(len(top_candidates), n_rows * n_cols):
        r_idx = j // n_cols
        c_idx = j % n_cols
        axes[r_idx, c_idx].axis("off")

    plt.tight_layout()
    plt.show()
