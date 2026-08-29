import os
import base64
import io
import numpy as np
from PIL import Image
import model


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


def find_keyframe_image_path(base_kf_dir, v_id, kf_name):
    """
    Tự động tìm đường dẫn file ảnh keyframe trong các thư mục Keyframes_Lxx, Keyframes_Lxx_a, _b...
    """
    prefix = v_id.split('_')[0]
    for sub_tag in ["", "_a", "_b", "_c", "_d", "_e"]:
        candidate = os.path.join(base_kf_dir, f"Keyframes_{prefix}{sub_tag}", "keyframes", v_id, kf_name)
        if os.path.exists(candidate):
            return candidate
    return None


def create_interactive_card(rank, v_id, initial_kf_name, score, global_map, metadata, base_kf_dir):
    """
    Tạo Card trực quan tương tác có các nút ◀ Prev và Next ▶ để lùi / tiến keyframe theo thời gian thực.
    """
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output

    meta = metadata.get(v_id, {})
    yt_url = meta.get("watch_url", "")
    title = meta.get("title", "No Title")

    # State lưu trữ keyframe hiện tại
    state = {
        "kf_int": int(initial_kf_name.replace(".jpg", ""))
    }

    card_output = widgets.Output()

    btn_prev_5 = widgets.Button(description="⏪ -5", layout=widgets.Layout(width="65px", height="34px"))
    btn_prev_1 = widgets.Button(description="◀ Prev", button_style="info", layout=widgets.Layout(width="85px", height="34px"))
    lbl_frame_indicator = widgets.HTML(layout=widgets.Layout(margin="0 15px"))
    btn_next_1 = widgets.Button(description="Next ▶", button_style="info", layout=widgets.Layout(width="85px", height="34px"))
    btn_next_5 = widgets.Button(description="+5 ⏩", layout=widgets.Layout(width="65px", height="34px"))

    nav_toolbar = widgets.HBox(
        [btn_prev_5, btn_prev_1, lbl_frame_indicator, btn_next_1, btn_next_5],
        layout=widgets.Layout(justify_content="center", align_items="center", margin="12px 0")
    )

    def update_view():
        kf_int = state["kf_int"]
        curr_kf_name = f"{kf_int:03d}.jpg"
        curr_key = f"{v_id}/{curr_kf_name}"

        map_info = global_map.get(curr_key, {})
        if isinstance(map_info, dict):
            frame_idx = map_info.get("frame_idx", 0)
            pts_time = int(map_info.get("pts_time", 0.0))
        else:
            frame_idx = map_info if map_info else 0
            pts_time = 0

        timestamp_link = f"{yt_url}&t={pts_time}s" if yt_url else "#"
        img_path = find_keyframe_image_path(base_kf_dir, v_id, curr_kf_name)

        lbl_frame_indicator.value = f"""
        <span style="font-size: 16px; font-weight: bold; color: #38bdf8; background: #0f172a; padding: 6px 14px; border-radius: 6px; border: 1px solid #334155;">
            {v_id} / <code>{curr_kf_name}</code>
        </span>
        """

        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
                b64_str = image_to_base64(img, max_width=950)
                img_html = f"""
                <div style="text-align: center; margin: 10px 0;">
                    <img src="data:image/jpeg;base64,{b64_str}" 
                         style="max-width: 95%; width: 850px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.4); border: 2px solid #3b82f6;" />
                </div>
                """
            except Exception as e:
                img_html = f"<div style='padding: 20px; background: #333; color: #ff6b6b;'>Lỗi đọc ảnh: {e}</div>"
        else:
            img_html = f"""
            <div style="text-align: center; padding: 40px; background: #2a2a2a; color: #aaa; border-radius: 10px; font-size: 16px;">
                ⚠️ Không tìm thấy file <code>{curr_kf_name}</code> trên đĩa.
            </div>
            """

        card_html = f"""
        <div style="margin: 15px auto; max-width: 950px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            
            <!-- 1. KHUNG TIÊU ĐỀ & RANK -->
            <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px 12px 0 0; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; border-left: 6px solid #3b82f6;">
                <span style="font-size: 22px; font-weight: 800; color: #60a5fa;">
                    🏆 TOP {rank} <span style="font-size: 16px; font-weight: normal; color: #cbd5e1;">(Score: <b style="color: #34d399;">{score:.4f}</b>)</span>
                </span>
                <span style="background: #1e3a8a; color: #93c5fd; padding: 4px 14px; border-radius: 20px; font-size: 15px; font-weight: 600;">
                    {v_id} / {curr_kf_name}
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
                    <div><b style="color: #94a3b8;">🖼️ Frame ID:</b> <b style="color: #fbbf24; font-size: 16px;">{frame_idx}</b> (<code>{curr_kf_name}</code>)</div>
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
        </div>
        """

        with card_output:
            clear_output(wait=True)
            display(HTML(card_html))

    def on_step(delta):
        state["kf_int"] = max(1, state["kf_int"] + delta)
        update_view()

    btn_prev_5.on_click(lambda _: on_step(-5))
    btn_prev_1.on_click(lambda _: on_step(-1))
    btn_next_1.on_click(lambda _: on_step(1))
    btn_next_5.on_click(lambda _: on_step(5))

    update_view()

    separator = HTML("""<div style="margin: 35px auto 25px auto; height: 3px; max-width: 950px; background: linear-gradient(90deg, transparent, #3b82f6, #8b5cf6, transparent);"></div>""")

    card_widget = widgets.VBox([
        nav_toolbar,
        card_output,
        widgets.HTML(separator.data)
    ])
    return card_widget


def inspect_query(
    processor,
    model_obj,
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
    Tìm kiếm câu query Tiếng Anh (dạng string) và hiển thị kết quả trực quan dạng thẻ lớn kèm nút Prev/Next Frame.
    """
    query_en = str(query_en).strip()
    if not query_en:
        print("⚠️ Query rỗng, vui lòng nhập chuỗi mô tả tiếng Anh.")
        return

    print(f"🔎 Đang tìm kiếm: \"{query_en}\"")

    # 1. Encode text query đơn
    query_vec = model.encode_text_query(
        processor,
        model_obj,
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

    # 4. Hiển thị từng Card có nút chuyển Frame linh hoạt
    try:
        import ipywidgets as widgets
        from IPython.display import display
        use_widgets = True
    except ImportError:
        use_widgets = False

    for i, cand in enumerate(top_candidates):
        v_id = cand["video"]
        kf_name = cand["kf_name"]
        score = cand["score"]

        if use_widgets and show_html:
            card_widget = create_interactive_card(
                rank=i + 1,
                v_id=v_id,
                initial_kf_name=kf_name,
                score=score,
                global_map=global_map,
                metadata=metadata,
                base_kf_dir=base_kf_dir,
            )
            display(card_widget)
        else:
            frame_idx = cand["frame_idx"]
            pts_time = int(cand["pts_time"])
            meta = metadata.get(v_id, {})
            yt_url = meta.get("watch_url", "")
            timestamp_link = f"{yt_url}&t={pts_time}s" if yt_url else "#"
            print(f"Top {i+1}: {v_id}/{kf_name} | Score: {score:.4f} | Frame: {frame_idx} | Time: {pts_time}s | {timestamp_link}")
