import os
import base64
import io
import numpy as np
from PIL import Image
from model import encode_text_query


def image_to_base64(img, max_width=900):
    """
    Chuyển đổi PIL Image sang chuỗi Base64 để hiển thị trực tiếp trong HTML với kích thước lớn.
    """
    w, h = img.size
    if w > max_width:
        ratio = max_width / float(w)
        img = img.resize((max_width, int(h * ratio)), Image.Resampling.LANCZOS)

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def inspect_query(
    processor,
    model,
    index,
    manifest,
    global_map,
    metadata,
    query_en: str,
    top_n=6,
    base_kf_dir="/kaggle/input/datasets/nguynhuyds/aic-dataset",
    vector_search_top_k=200,
    max_length=64,
    device="cuda",
    show_html=True,
):
    """
    Tìm kiếm câu query Tiếng Anh (dạng string) và hiển thị kết quả trực quan dạng thẻ lớn.
    """
    query_en = str(query_en).strip()
    if not query_en:
        print("⚠️ Query rỗng, vui lòng nhập chuỗi mô tả tiếng Anh.")
        return

    print(f"🔎 Đang tìm kiếm: \"{query_en}\"")

    # 1. Encode text query đơn
    query_vec = encode_text_query(
        processor,
        model,
        query_en,
        device=device,
        max_length=max_length,
        truncation=True,
    )

    # 2. Vector Search (Lấy top candidates)
    scores, indices = index.search(query_vec, vector_search_top_k)
    scores, indices = scores[0], indices[0]

    # 3. Gom nhóm kết quả (tối đa 2 frame / video)
    candidates = []
    seen_videos = {}

    for score, idx in zip(scores, indices):
        kf_key = manifest[idx]
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

    top_candidates = candidates[:top_n]

    if not top_candidates:
        print("❌ Không tìm thấy kết quả phù hợp.")
        return

    # 4. Hiển thị tuần tự: [ẢNH TO] -> [CHI TIẾT RÕ RÀNG] -> [PHÂN CÁCH]
    try:
        from IPython.display import display, HTML
        has_ipython = True
    except ImportError:
        has_ipython = False

    for i, cand in enumerate(top_candidates):
        v_id = cand["video"]
        kf_name = cand["kf_name"]
        frame_idx = cand["frame_idx"]
        pts_time = int(cand["pts_time"])
        score = cand["score"]

        meta = metadata.get(v_id, {})
        yt_url = meta.get("watch_url", "")
        title = meta.get("title", "No Title")
        timestamp_link = f"{yt_url}&t={pts_time}s" if yt_url else "#"

        # Tự động tìm đúng thư mục gốc (Keyframes_L26_a, _b, _c, ...)
        prefix = v_id.split('_')[0]
        img_path = None
        for sub_tag in ["", "_a", "_b", "_c", "_d", "_e"]:
            candidate = os.path.join(base_kf_dir, f"Keyframes_{prefix}{sub_tag}", "keyframes", v_id, kf_name)
            if os.path.exists(candidate):
                img_path = candidate
                break

        img_html = ""
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                b64_str = image_to_base64(img, max_width=950)
                img_html = f"""
                <div style="text-align: center; margin: 15px 0;">
                    <img src="data:image/jpeg;base64,{b64_str}" 
                         style="max-width: 95%; width: 850px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.3); border: 2px solid #3b82f6;" />
                </div>
                """
            except Exception as e:
                img_html = f"<div style='padding: 20px; background: #333; color: #ff6b6b;'>Lỗi đọc ảnh: {e}</div>"
        else:
            img_html = """
            <div style="text-align: center; padding: 40px; background: #2a2a2a; color: #aaa; border-radius: 10px; font-size: 18px;">
                ⚠️ Không tìm thấy file ảnh keyframe trên đĩa.
            </div>
            """

        card_html = f"""
        <div style="margin: 25px auto; max-width: 950px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            
            <!-- 1. KHUNG TIÊU ĐỀ & RANK -->
            <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px 12px 0 0; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; border-left: 6px solid #3b82f6;">
                <span style="font-size: 22px; font-weight: 800; color: #60a5fa;">
                    🏆 TOP {i+1} <span style="font-size: 16px; font-weight: normal; color: #cbd5e1;">(Score: <b style="color: #34d399;">{score:.4f}</b>)</span>
                </span>
                <span style="background: #1e3a8a; color: #93c5fd; padding: 4px 14px; border-radius: 20px; font-size: 15px; font-weight: 600;">
                    {v_id} / {kf_name}
                </span>
            </div>

            <!-- 2. ẢNH KEYFRAME LỚN TRỰC QUAN -->
            <div style="background: #111827; padding: 10px 0;">
                {img_html}
            </div>

            <!-- 3. KHUNG THÔNG TIN CHI TIẾT -->
            <div style="background: #1e293b; color: #f8fafc; border-radius: 0 0 12px 12px; padding: 18px 24px; font-size: 16px; line-height: 1.6; border: 1px solid #334155; border-top: none;">
                <div style="margin-bottom: 8px;">
                    <b style="color: #94a3b8;">📌 Video Title:</b> <span style="color: #f1f5f9; font-weight: 600; font-size: 17px;">{title}</span>
                </div>
                <div style="margin-bottom: 12px; display: flex; gap: 20px; flex-wrap: wrap;">
                    <div><b style="color: #94a3b8;">🎬 Video ID:</b> <code style="background: #0f172a; color: #38bdf8; padding: 3px 8px; border-radius: 4px; font-size: 15px;">{v_id}</code></div>
                    <div><b style="color: #94a3b8;">🖼️ Frame ID:</b> <b style="color: #fbbf24; font-size: 16px;">{frame_idx}</b> (<code>{kf_name}</code>)</div>
                    <div><b style="color: #94a3b8;">⏱️ Timestamp:</b> <b style="color: #a78bfa; font-size: 16px;">{pts_time}s</b></div>
                </div>

                <!-- 4. NÚT XEM YOUTUBE -->
                <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #334155;">
                    <a href="{timestamp_link}" target="_blank" 
                       style="display: inline-block; background: #dc2626; color: #ffffff; text-decoration: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; font-size: 15px; box-shadow: 0 4px 12px rgba(220,38,38,0.4);">
                        ▶ Xem Video trên YouTube tại {pts_time}s
                    </a>
                </div>
            </div>

            <!-- 5. ĐƯỜNG KẺ PHÂN CÁCH LỚN -->
            <div style="margin: 40px auto 30px auto; height: 3px; background: linear-gradient(90deg, transparent, #3b82f6, #8b5cf6, transparent);"></div>
        </div>
        """

        if has_ipython and show_html:
            display(HTML(card_html))
        else:
            print(f"Top {i+1}: {v_id}/{kf_name} | Score: {score:.4f} | Frame: {frame_idx} | Time: {pts_time}s | {timestamp_link}")
