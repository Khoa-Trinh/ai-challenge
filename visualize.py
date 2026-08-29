import os
import base64
import io
import numpy as np
from PIL import Image
from model import encode_text_queries


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


def get_keyframe_image_html(base_kf_dir, v_id, kf_name, max_width=850):
    """
    Tìm và sinh mã HTML hiển thị ảnh keyframe.
    """
    prefix = v_id.split('_')[0]
    img_path = None
    for sub_tag in ["", "_a", "_b", "_c", "_d", "_e"]:
        candidate = os.path.join(base_kf_dir, f"Keyframes_{prefix}{sub_tag}", "keyframes", v_id, kf_name)
        if os.path.exists(candidate):
            img_path = candidate
            break

    if img_path and os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            b64_str = image_to_base64(img, max_width=max_width)
            return f"""
            <div style="text-align: center; margin: 10px 0;">
                <img src="data:image/jpeg;base64,{b64_str}" 
                     style="max-width: 98%; width: {max_width}px; border-radius: 10px; box-shadow: 0 6px 20px rgba(0,0,0,0.35); border: 2px solid #3b82f6;" />
            </div>
            """
        except Exception as e:
            return f"<div style='padding: 15px; background: #222; color: #ff6b6b;'>Lỗi đọc ảnh: {e}</div>"
    else:
        return f"""
        <div style="text-align: center; padding: 30px; background: #1e293b; color: #94a3b8; border-radius: 8px; font-size: 15px;">
            ⚠️ Không tìm thấy ảnh keyframe: <code>{v_id}/{kf_name}</code>
        </div>
        """


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
    MODE 1: SYNONYMS / SINGLE SCENE SEARCH
    Tìm kiếm và hiển thị từng kết quả: Ảnh Keyframe LỚN -> Chi tiết rõ ràng -> Đường kẻ phân cách.
    """
    print(f"🔎 [Synonyms Mode] Đang tìm kiếm cho: '{query_vi}'...")

    # 1. Encode text queries với padding và truncation
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

        img_html = get_keyframe_image_html(base_kf_dir, v_id, kf_name, max_width=850)

        card_html = f"""
        <div style="margin: 25px auto; max-width: 950px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px 12px 0 0; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; border-left: 6px solid #3b82f6;">
                <span style="font-size: 22px; font-weight: 800; color: #60a5fa;">
                    🏆 TOP {i+1} <span style="font-size: 16px; font-weight: normal; color: #cbd5e1;">(Score: <b style="color: #34d399;">{score:.4f}</b>)</span>
                </span>
                <span style="background: #1e3a8a; color: #93c5fd; padding: 4px 14px; border-radius: 20px; font-size: 15px; font-weight: 600;">
                    {v_id} / {kf_name}
                </span>
            </div>

            <div style="background: #111827; padding: 10px 0;">
                {img_html}
            </div>

            <div style="background: #1e293b; color: #f8fafc; border-radius: 0 0 12px 12px; padding: 18px 24px; font-size: 16px; line-height: 1.6; border: 1px solid #334155; border-top: none;">
                <div style="margin-bottom: 8px;">
                    <b style="color: #94a3b8;">📌 Video Title:</b> <span style="color: #f1f5f9; font-weight: 600; font-size: 17px;">{title}</span>
                </div>
                <div style="margin-bottom: 12px; display: flex; gap: 20px; flex-wrap: wrap;">
                    <div><b style="color: #94a3b8;">🎬 Video ID:</b> <code style="background: #0f172a; color: #38bdf8; padding: 3px 8px; border-radius: 4px; font-size: 15px;">{v_id}</code></div>
                    <div><b style="color: #94a3b8;">🖼️ Frame ID:</b> <b style="color: #fbbf24; font-size: 16px;">{frame_idx}</b> (<code>{kf_name}</code>)</div>
                    <div><b style="color: #94a3b8;">⏱️ Timestamp:</b> <b style="color: #a78bfa; font-size: 16px;">{pts_time}s</b></div>
                </div>

                <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #334155;">
                    <a href="{timestamp_link}" target="_blank" 
                       style="display: inline-block; background: #dc2626; color: #ffffff; text-decoration: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; font-size: 15px; box-shadow: 0 4px 12px rgba(220,38,38,0.4);">
                        ▶ Xem Video trên YouTube tại {pts_time}s
                    </a>
                </div>
            </div>

            <div style="margin: 40px auto 30px auto; height: 3px; background: linear-gradient(90deg, transparent, #3b82f6, #8b5cf6, transparent);"></div>
        </div>
        """

        if has_ipython and show_html:
            display(HTML(card_html))
        else:
            print(f"Top {i+1}: {v_id}/{kf_name} | Score: {score:.4f} | Frame: {frame_idx} | Time: {pts_time}s | {timestamp_link}")


def inspect_sequential_query(
    processor,
    model,
    index,
    manifest,
    global_map,
    metadata,
    steps_en_list,
    steps_vi_list=None,
    top_n=6,
    base_kf_dir="/kaggle/input/datasets/nguynhuyds/aic-dataset",
    vector_search_top_k=300,
    max_time_gap=240,
    max_length=64,
    device="cuda",
    show_html=True,
):
    """
    MODE 2: SEQUENTIAL ACTIONS / TIMELINE SEARCH (với hỗ trợ Synonyms cho từng bước)
    - Mỗi phần tử trong steps_en_list có thể là chuỗi đơn hoặc 1 danh sách các câu đồng nghĩa cho riêng bước đó.
    - Tìm kiếm độc lập từng bước trên FAISS -> lọc chuỗi thời gian (t1 <= t2 <= ...) -> hiển thị song song.
    """
    num_steps = len(steps_en_list)
    if num_steps < 2:
        print("⚠️ Chế độ Sequential cần ít nhất 2 bước hành động. Chuyển sang Synonyms search.")
        first_step = steps_en_list[0] if steps_en_list else []
        if isinstance(first_step, str):
            first_step = [first_step]
        return inspect_query(
            processor=processor,
            model=model,
            index=index,
            manifest=manifest,
            global_map=global_map,
            metadata=metadata,
            query_vi=" ".join(steps_vi_list[0] if (steps_vi_list and isinstance(steps_vi_list[0], list)) else (steps_vi_list or first_step)),
            query_en_list=first_step,
            top_n=top_n,
            base_kf_dir=base_kf_dir,
            vector_search_top_k=vector_search_top_k,
            max_length=max_length,
            device=device,
            show_html=show_html,
        )

    print(f"🔎 [Sequential + Synonyms Mode] Đang tìm kiếm chuỗi {num_steps} bước hành động theo thời gian...")

    # 1. Tìm kiếm FAISS riêng cho từng bước (kết hợp Synonyms nếu có)
    step_results = []  # step_results[step_idx] = {v_id: [candidates]}
    for step_idx, step_query in enumerate(steps_en_list):
        # Hỗ trợ cả chuỗi đơn và danh sách đồng nghĩa
        if isinstance(step_query, str):
            step_query_list = [step_query]
        else:
            step_query_list = [str(q).strip() for q in step_query if str(q).strip()]

        query_vec = encode_text_queries(
            processor,
            model,
            step_query_list,
            device=device,
            max_length=max_length,
            truncation=True,
        )
        scores, indices = index.search(query_vec, vector_search_top_k)
        scores, indices = scores[0], indices[0]

        video_cands = {}
        for score, idx in zip(scores, indices):
            kf_key = manifest[idx]
            v_id, kf_name = kf_key.split("/")

            map_info = global_map.get(kf_key, {})
            if isinstance(map_info, dict):
                frame_idx = map_info.get("frame_idx", 0)
                pts_time = map_info.get("pts_time", 0.0)
            else:
                frame_idx = map_info
                pts_time = 0.0

            item = {
                "kf_key": kf_key,
                "video": v_id,
                "kf_name": kf_name,
                "frame_idx": frame_idx,
                "pts_time": pts_time,
                "score": float(score),
            }

            if v_id not in video_cands:
                video_cands[v_id] = []
            video_cands[v_id].append(item)

        step_results.append(video_cands)

    # 2. Ghép nối và kiểm tra tính nhất quán thời gian (Temporal Alignment)
    matched_sequences = []

    for v_id, cands_step0 in step_results[0].items():
        has_all_steps = all(v_id in step_results[s] for s in range(1, num_steps))

        best_combo = None
        best_combo_score = -1.0

        if num_steps == 2 and v_id in step_results[1]:
            for c0 in cands_step0:
                for c1 in step_results[1][v_id]:
                    # Điều kiện thời gian: t0 <= t1 và không cách nhau quá max_time_gap
                    time_diff = c1["pts_time"] - c0["pts_time"]
                    if 0 <= time_diff <= max_time_gap:
                        combo_score = (c0["score"] + c1["score"]) / 2.0 + 0.05  # Thưởng điểm thứ tự thời gian
                    else:
                        combo_score = (c0["score"] + c1["score"]) / 2.0  # Không thưởng nếu sai thứ tự

                    if combo_score > best_combo_score:
                        best_combo_score = combo_score
                        best_combo = [c0, c1]
        elif has_all_steps:
            # Hỗ trợ 3+ bước
            c0 = cands_step0[0]
            combo = [c0]
            curr_score = c0["score"]
            valid_order = True
            for s in range(1, num_steps):
                cs = step_results[s][v_id][0]
                if cs["pts_time"] < combo[-1]["pts_time"]:
                    valid_order = False
                combo.append(cs)
                curr_score += cs["score"]
            combo_score = (curr_score / num_steps) + (0.05 if valid_order else 0.0)
            best_combo = combo
            best_combo_score = combo_score
        else:
            # Video chỉ khớp bước 1 (cho điểm thấp hơn)
            c0 = cands_step0[0]
            best_combo = [c0] + [None] * (num_steps - 1)
            best_combo_score = c0["score"] * 0.7

        if best_combo is not None:
            matched_sequences.append({
                "video": v_id,
                "score": best_combo_score,
                "steps": best_combo,
                "has_all_steps": has_all_steps,
            })

    matched_sequences.sort(key=lambda x: x["score"], reverse=True)
    top_sequences = matched_sequences[:top_n]

    if not top_sequences:
        print("❌ Không tìm thấy chuỗi video phù hợp.")
        return

    try:
        from IPython.display import display, HTML
        has_ipython = True
    except ImportError:
        has_ipython = False

    for i, seq in enumerate(top_sequences):
        v_id = seq["video"]
        score = seq["score"]
        steps_data = seq["steps"]

        meta = metadata.get(v_id, {})
        yt_url = meta.get("watch_url", "")
        title = meta.get("title", "No Title")

        # Sinh HTML cho từng bước hành động
        steps_html = ""
        first_time = 0
        for s_idx, cand in enumerate(steps_data):
            # Lấy mô tả hiển thị
            raw_vi = steps_vi_list[s_idx] if steps_vi_list and s_idx < len(steps_vi_list) else ""
            if isinstance(raw_vi, list):
                step_title_vi = " / ".join(raw_vi)
            else:
                step_title_vi = str(raw_vi)

            raw_en = steps_en_list[s_idx] if s_idx < len(steps_en_list) else ""
            if isinstance(raw_en, list):
                step_en = " / ".join(raw_en)
            else:
                step_en = str(raw_en)

            if cand is not None:
                kf_name = cand["kf_name"]
                frame_idx = cand["frame_idx"]
                pts_time = int(cand["pts_time"])
                if s_idx == 0:
                    first_time = pts_time
                cand_score = cand["score"]
                img_html = get_keyframe_image_html(base_kf_dir, v_id, kf_name, max_width=420)
                link = f"{yt_url}&t={pts_time}s" if yt_url else "#"

                step_card = f"""
                <div style="flex: 1; min-width: 320px; background: #0f172a; border-radius: 10px; padding: 14px; border: 1px solid #334155;">
                    <div style="font-size: 15px; font-weight: bold; color: #38bdf8; margin-bottom: 6px;">
                        📍 Bước {s_idx + 1}: <span style="font-weight: normal; color: #cbd5e1; font-size: 13px;">{step_title_vi or step_en}</span>
                    </div>
                    {img_html}
                    <div style="margin-top: 8px; font-size: 14px; color: #cbd5e1; line-height: 1.5;">
                        <div>Frame: <b style="color: #fbbf24;">{frame_idx}</b> | Keyframe: <code>{kf_name}</code></div>
                        <div>Timestamp: <b style="color: #a78bfa;">{pts_time}s</b> | Sim: <b style="color: #34d399;">{cand_score:.4f}</b></div>
                        <div style="margin-top: 8px;">
                            <a href="{link}" target="_blank" style="color: #f87171; font-weight: bold; text-decoration: none;">
                                🔗 Xem YouTube tại {pts_time}s
                            </a>
                        </div>
                    </div>
                </div>
                """
            else:
                step_card = f"""
                <div style="flex: 1; min-width: 320px; background: #0f172a; border-radius: 10px; padding: 20px; border: 1px dashed #475569; text-align: center; color: #64748b;">
                    <div style="font-size: 15px; font-weight: bold; color: #94a3b8; margin-bottom: 6px;">📍 Bước {s_idx + 1}</div>
                    <div style="padding: 40px 0;">⚠️ Không tìm thấy frame khớp cho bước này trong video</div>
                </div>
                """
            steps_html += step_card

        full_yt_link = f"{yt_url}&t={first_time}s" if yt_url else "#"

        card_html = f"""
        <div style="margin: 30px auto; max-width: 980px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #312e81, #1e1b4b); border-radius: 12px 12px 0 0; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; border-left: 6px solid #818cf8;">
                <span style="font-size: 22px; font-weight: 800; color: #a5b4fc;">
                    🎬 TOP {i+1} SEQUENCE <span style="font-size: 16px; font-weight: normal; color: #e0e7ff;">(Seq Score: <b style="color: #34d399;">{score:.4f}</b>)</span>
                </span>
                <span style="background: #4338ca; color: #e0e7ff; padding: 4px 14px; border-radius: 20px; font-size: 15px; font-weight: 600;">
                    Video: {v_id}
                </span>
            </div>

            <!-- Video Info -->
            <div style="background: #1e293b; color: #f8fafc; padding: 12px 20px; font-size: 15px; border-left: 1px solid #334155; border-right: 1px solid #334155;">
                <b>📌 Tiêu đề:</b> {title}
            </div>

            <!-- Steps side by side -->
            <div style="background: #111827; padding: 16px; display: flex; gap: 16px; flex-wrap: wrap; border: 1px solid #334155; border-top: none; border-radius: 0 0 12px 12px;">
                {steps_html}
            </div>

            <!-- Separator -->
            <div style="margin: 40px auto 30px auto; height: 3px; background: linear-gradient(90deg, transparent, #818cf8, #ec4899, transparent);"></div>
        </div>
        """

        if has_ipython and show_html:
            display(HTML(card_html))
        else:
            print(f"Top {i+1} Seq: {v_id} | Combined Score: {score:.4f} | {full_yt_link}")
