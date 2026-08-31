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


def extract_query_entities(query_en):
    """
    Trích xuất danh sách các thực thể / từ khóa vật thể từ câu query tiếng Anh để đối soát với OpenImages / Global Objects.
    """
    import re
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', query_en.lower())
    words = [w for w in cleaned.split() if len(w) > 2]
    
    # Từ dừng (stopwords) phổ biến trong mô tả hình ảnh
    stopwords = {
        "the", "and", "with", "this", "that", "there", "from", "into", "onto", "over", 
        "under", "near", "next", "behind", "front", "side", "view", "scene", "video", 
        "clip", "frame", "showing", "shows", "look", "looks", "looking", "take", "takes", 
        "taking", "color", "colors", "colored", "background", "foreground", "photo", 
        "picture", "image", "first", "last", "then", "after", "before", "many", "some"
    }
    
    # Từ đồng nghĩa / ánh xạ sang OpenImages Entities
    synonym_map = {
        "cyclist": ["person", "bicycle", "vehicle"],
        "cyclists": ["person", "bicycle", "vehicle"],
        "biker": ["person", "motorcycle", "bicycle", "vehicle"],
        "bikers": ["person", "motorcycle", "bicycle", "vehicle"],
        "bicycle": ["bicycle", "vehicle"],
        "bicycles": ["bicycle", "vehicle"],
        "bike": ["bicycle", "motorcycle", "vehicle"],
        "bikes": ["bicycle", "motorcycle", "vehicle"],
        "motorbike": ["motorcycle", "vehicle"],
        "motorcycle": ["motorcycle", "vehicle"],
        "car": ["car", "land vehicle", "vehicle"],
        "cars": ["car", "land vehicle", "vehicle"],
        "automobile": ["car", "land vehicle", "vehicle"],
        "boat": ["boat", "watercraft", "vehicle"],
        "boats": ["boat", "watercraft", "vehicle"],
        "ship": ["boat", "watercraft", "vehicle"],
        "chef": ["person", "clothing"],
        "cook": ["person", "food", "table"],
        "man": ["person", "man"],
        "men": ["person", "man"],
        "woman": ["person", "woman"],
        "women": ["person", "woman"],
        "boy": ["person", "man", "boy"],
        "girl": ["person", "woman", "girl"],
        "child": ["person", "boy", "girl"],
        "children": ["person", "boy", "girl"],
        "people": ["person", "man", "woman"],
        "fish": ["fish", "seafood", "animal"],
        "fishes": ["fish", "seafood", "animal"],
        "rhino": ["rhinoceros", "animal"],
        "rhinoceros": ["rhinoceros", "animal"],
        "monkey": ["monkey", "animal"],
        "monkeys": ["monkey", "animal"],
        "building": ["building", "skyscraper", "tower"],
        "buildings": ["building", "skyscraper", "tower"],
        "house": ["building", "house"],
        "tower": ["tower", "building", "skyscraper"],
        "bridge": ["bridge", "building"],
        "tree": ["tree", "plant"],
        "trees": ["tree", "plant"],
        "plant": ["plant", "flower", "tree"],
        "plants": ["plant", "flower", "tree"],
        "table": ["table", "furniture", "desk"],
        "chair": ["chair", "furniture"],
        "balloon": ["balloon", "toy"],
        "balloons": ["balloon", "toy"],
        "lantern": ["lantern", "lamp", "lighting"],
        "lamp": ["lamp", "street light", "lighting"],
        "drum": ["drum", "musical instrument"],
        "flag": ["flag", "poster"],
        "hat": ["hat", "fashion accessory", "clothing"],
        "helmet": ["helmet", "fashion accessory", "clothing"],
    }
    
    query_entities = set()
    for w in words:
        if w not in stopwords:
            query_entities.add(w)
            # Thử bỏ s hoặc es số nhiều
            if w.endswith("es") and len(w) > 4:
                query_entities.add(w[:-2])
            elif w.endswith("s") and len(w) > 3:
                query_entities.add(w[:-1])
                
            if w in synonym_map:
                for mapped in synonym_map[w]:
                    query_entities.add(mapped)
                    
    return query_entities


def calculate_object_match_score(frame_objs, query_entities):
    """
    Tính điểm khớp vật thể giữa frame detections và query entities.
    """
    if not frame_objs or not query_entities:
        return 0.0, {}
        
    matched = {}
    for entity, conf in frame_objs.items():
        entity_lower = entity.lower()
        # Kiểm tra exact match hoặc substring match
        for q_ent in query_entities:
            if q_ent == entity_lower or q_ent in entity_lower.split():
                matched[entity] = conf
                break
                
    if not matched:
        return 0.0, {}
        
    score = sum(matched.values())
    return score, matched


def create_interactive_card(rank, v_id, initial_kf_name, score, siglip_score, matched_objs, global_map, metadata, global_objects, base_kf_dir, matched_transcript=None):
    """
    Tạo Card trực quan tương tác có các nút ◀ Prev và Next ▶ để lùi / tiến keyframe theo thời gian thực,
    đồng thời hiển thị thông tin Object Detections và Video Speech Transcripts.
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

        # Hỗ trợ cả keyframes_map trong cấu trúc mới
        kf_map = global_map.get("keyframes_map", global_map) if isinstance(global_map, dict) else {}
        map_info = kf_map.get(curr_key, {})
        if isinstance(map_info, dict):
            frame_idx = map_info.get("frame_idx", 0)
            pts_time = int(map_info.get("pts_time", 0.0))
        else:
            frame_idx = map_info if map_info else 0
            pts_time = 0

        # Nếu có mốc giây từ transcript match thì ưu tiên
        target_yt_time = pts_time
        if matched_transcript and "start_sec" in matched_transcript:
            target_yt_time = int(matched_transcript["start_sec"])

        timestamp_link = f"{yt_url}&t={target_yt_time}s" if yt_url else "#"
        img_path = find_keyframe_image_path(base_kf_dir, v_id, curr_kf_name)

        # Lấy Objects của frame hiện tại
        frame_objs = global_objects.get(curr_key, {}) if global_objects else {}
        
        objects_html_list = []
        if frame_objs:
            # Sắp xếp theo score giảm dần
            sorted_objs = sorted(frame_objs.items(), key=lambda x: x[1], reverse=True)[:8]
            for ent, conf in sorted_objs:
                is_matched = ent in matched_objs or ent.lower() in [m.lower() for m in matched_objs]
                if is_matched:
                    badge_style = "background: #065f46; color: #6ee7b7; border: 1px solid #10b981; font-weight: bold;"
                    star = "🎯 "
                else:
                    badge_style = "background: #1e293b; color: #94a3b8; border: 1px solid #334155;"
                    star = ""
                objects_html_list.append(
                    f"""<span style="display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 13px; margin: 2px; {badge_style}">
                        {star}{ent} ({conf:.2f})
                    </span>"""
                )
            objects_display = "".join(objects_html_list)
        else:
            objects_display = """<span style="color: #64748b; font-size: 13px; font-style: italic;">Không có detections</span>"""

        transcript_display_html = ""
        if matched_transcript:
            t_sec = matched_transcript.get("start_sec", 0)
            t_text = matched_transcript.get("text", "")
            transcript_display_html = f"""
            <div style="margin-bottom: 12px; background: #422006; border: 1px solid #eab308; border-radius: 8px; padding: 10px 14px; color: #fef08a;">
                <div style="font-weight: 700; font-size: 14px; margin-bottom: 4px;">🎙️ Khớp Lời Thoại / Transcript tại giây {t_sec}s:</div>
                <div style="font-size: 15px; font-style: italic; color: #ffffff;">"{t_text}"</div>
            </div>
            """

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

        score_details = f"Score: <b style='color: #34d399;'>{score:.4f}</b>"
        if siglip_score is not None and abs(score - siglip_score) > 1e-4:
            score_details += f" <span style='font-size: 13px; color: #94a3b8;'>(SigLIP: {siglip_score:.4f})</span>"

        card_html = f"""
        <div style="margin: 15px auto; max-width: 950px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            
            <!-- 1. KHUNG TIÊU ĐỀ & RANK -->
            <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border-radius: 12px 12px 0 0; padding: 14px 20px; display: flex; justify-content: space-between; align-items: center; border-left: 6px solid #3b82f6;">
                <span style="font-size: 22px; font-weight: 800; color: #60a5fa;">
                    🏆 TOP {rank} <span style="font-size: 16px; font-weight: normal; color: #cbd5e1;">({score_details})</span>
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
                
                {transcript_display_html}

                <div style="margin-bottom: 8px;">
                    <b style="color: #94a3b8;">📌 Video Title:</b> <span style="color: #f1f5f9; font-weight: 600; font-size: 17px;">{title}</span>
                </div>
                <div style="margin-bottom: 10px; display: flex; gap: 20px; flex-wrap: wrap;">
                    <div><b style="color: #94a3b8;">🎬 Video ID:</b> <code style="background: #0f172a; color: #38bdf8; padding: 3px 8px; border-radius: 4px; font-size: 15px;">{v_id}</code></div>
                    <div><b style="color: #94a3b8;">🖼️ Frame ID:</b> <b style="color: #fbbf24; font-size: 16px;">{frame_idx}</b> (<code>{curr_kf_name}</code>)</div>
                    <div><b style="color: #94a3b8;">⏱️ Timestamp:</b> <b style="color: #a78bfa; font-size: 16px;">{pts_time}s</b></div>
                </div>

                <!-- 4. KHUNG OBJECT DETECTIONS -->
                <div style="margin-bottom: 12px; background: #0f172a; padding: 10px 14px; border-radius: 8px; border: 1px solid #334155;">
                    <b style="color: #38bdf8; font-size: 14px;">🏷️ Detected Objects:</b>
                    <div style="margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px;">
                        {objects_display}
                    </div>
                </div>

                <!-- 5. NÚT XEM YOUTUBE -->
                <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #334155;">
                    <a href="{timestamp_link}" target="_blank" 
                       style="display: inline-block; background: #dc2626; color: #ffffff; text-decoration: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; font-size: 15px; box-shadow: 0 4px 12px rgba(220,38,38,0.4);">
                        ▶ Xem Video trên YouTube tại {target_yt_time}s
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
    global_objects=None,
    inverted_objects=None,
    use_objects=True,
    object_weight=0.15,
    use_transcripts=True,
    transcript_weight=0.35,
    base_kf_dir="/kaggle/input/datasets/nguynhuyds/aic-dataset",
    vector_search_top_k=200,
    max_length=64,
    device="cuda",
    show_html=True,
):
    """
    Tìm kiếm đa phương thức (SigLIP Vector + YOLO-World Objects + Speech Transcripts),
    tự động Re-rank và hiển thị kết quả trực quan dạng thẻ lớn kèm nút Prev/Next Frame.
    """
    query_en = str(query_en).strip()
    if not query_en:
        print("⚠️ Query rỗng, vui lòng nhập chuỗi mô tả.")
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

    # 2. Vector Search (Lấy top candidates ban đầu từ SigLIP)
    scores, indices = index.search(query_vec, vector_search_top_k)
    scores, indices = scores[0], indices[0]

    # 3. Object-Aware Re-ranking
    query_entities = set()
    if use_objects and global_objects:
        query_entities = extract_query_entities(query_en)
        if query_entities:
            print(f"🎯 Thực thể đối soát Objects: {', '.join(sorted(query_entities))}")

    # 4. Transcript Search (Tìm kiếm phụ đề lời thoại)
    matched_transcripts_by_key = {}
    if use_transcripts and metadata:
        try:
            import transcript as ts_mod
            ts_matches = ts_mod.search_transcripts(metadata, query_en, global_map, max_results=8)
            if ts_matches:
                print(f"🎙️ Tìm thấy {len(ts_matches)} đoạn thoại khớp trong Speech Transcripts!")
                for m in ts_matches:
                    matched_transcripts_by_key[m["key"]] = m
        except Exception as e:
            pass

    kf_map = global_map.get("keyframes_map", global_map) if isinstance(global_map, dict) else {}

    raw_candidates = []
    seen_keys = set()

    # Thêm các candidates từ SigLIP Vector Search
    for score, idx in zip(scores, indices):
        kf_key = manifest[idx]
        seen_keys.add(kf_key)
        v_id, kf_name = kf_key.split("/")

        siglip_score = float(score)
        matched_objs = {}
        obj_match_score = 0.0

        if use_objects and global_objects and query_entities:
            frame_objs = global_objects.get(kf_key, {})
            obj_match_score, matched_objs = calculate_object_match_score(frame_objs, query_entities)

        # Transcript Bonus nếu khớp
        ts_match = matched_transcripts_by_key.get(kf_key)
        ts_bonus = (transcript_weight * ts_match["score"]) if ts_match else 0.0

        # Điểm kết hợp đa phương thức: SigLIP + Object + Transcript
        final_score = siglip_score + (object_weight * min(obj_match_score, 1.5)) + ts_bonus

        map_info = kf_map.get(kf_key, {})
        if isinstance(map_info, dict):
            frame_idx = map_info.get("frame_idx", 0)
            pts_time = map_info.get("pts_time", 0.0)
        else:
            frame_idx = map_info if map_info else 0
            pts_time = 0.0

        raw_candidates.append({
            "kf_key": kf_key,
            "video": v_id,
            "kf_name": kf_name,
            "frame_idx": frame_idx,
            "pts_time": pts_time,
            "siglip_score": siglip_score,
            "final_score": final_score,
            "matched_objs": matched_objs,
            "obj_match_score": obj_match_score,
            "matched_transcript": ts_match
        })

    # Nếu có Transcript Match chưa nằm trong top SigLIP, inject trực tiếp lên top
    for key, ts_match in matched_transcripts_by_key.items():
        if key not in seen_keys:
            v_id = ts_match["v_id"]
            kf_name = ts_match["kf_name"]
            map_info = kf_map.get(key, {})
            if isinstance(map_info, dict):
                frame_idx = map_info.get("frame_idx", 0)
                pts_time = map_info.get("pts_time", ts_match.get("start_sec", 0.0))
            else:
                frame_idx = map_info if map_info else 0
                pts_time = ts_match.get("start_sec", 0.0)

            # Điểm ưu tiên cao cho Transcript Exact Match
            injected_score = 0.25 + (transcript_weight * ts_match["score"])
            raw_candidates.append({
                "kf_key": key,
                "video": v_id,
                "kf_name": kf_name,
                "frame_idx": frame_idx,
                "pts_time": pts_time,
                "siglip_score": 0.20,
                "final_score": injected_score,
                "matched_objs": {},
                "obj_match_score": 0.0,
                "matched_transcript": ts_match
            })

    # Nếu có Inverted Objects Match với độ tự tin cao chưa nằm trong top SigLIP, inject vào candidates
    if use_objects and inverted_objects and query_entities:
        for q_ent in query_entities:
            matching_frames = inverted_objects.get(q_ent, {})
            if isinstance(matching_frames, dict):
                # Lấy các frame có confidence cao nhất cho object này
                for kf_k, conf in sorted(matching_frames.items(), key=lambda x: x[1], reverse=True)[:6]:
                    if kf_k not in seen_keys and conf >= 0.45:
                        seen_keys.add(kf_k)
                        if "/" in kf_k:
                            v_id, kf_name = kf_k.split("/", 1)
                        else:
                            continue

                        map_info = kf_map.get(kf_k, {})
                        if isinstance(map_info, dict):
                            frame_idx = map_info.get("frame_idx", 0)
                            pts_time = map_info.get("pts_time", 0.0)
                        else:
                            frame_idx = map_info if map_info else 0
                            pts_time = 0.0

                        obj_bonus = object_weight * min(float(conf), 1.5)
                        raw_candidates.append({
                            "kf_key": kf_k,
                            "video": v_id,
                            "kf_name": kf_name,
                            "frame_idx": frame_idx,
                            "pts_time": pts_time,
                            "siglip_score": 0.18,
                            "final_score": 0.18 + obj_bonus,
                            "matched_objs": {q_ent: float(conf)},
                            "obj_match_score": float(conf),
                            "matched_transcript": None
                        })

    # Sắp xếp lại theo điểm kết hợp Final Score
    raw_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # 5. Gom nhóm kết quả (tối đa 2 frame / video)
    candidates = []
    seen_videos = {}
    for cand in raw_candidates:
        v_id = cand["video"]
        if seen_videos.get(v_id, 0) < 2:
            seen_videos[v_id] = seen_videos.get(v_id, 0) + 1
            candidates.append(cand)

    top_candidates = candidates[:top_n]

    if not top_candidates:
        print("❌ Không tìm thấy kết quả phù hợp.")
        return

    # 6. Hiển thị từng Card có nút chuyển Frame linh hoạt & Badges
    try:
        import ipywidgets as widgets
        from IPython.display import display
        use_widgets = True
    except ImportError:
        use_widgets = False

    for i, cand in enumerate(top_candidates):
        v_id = cand["video"]
        kf_name = cand["kf_name"]
        final_score = cand["final_score"]
        siglip_score = cand["siglip_score"]
        matched_objs = cand["matched_objs"]
        matched_ts = cand.get("matched_transcript")

        if use_widgets and show_html:
            card_widget = create_interactive_card(
                rank=i + 1,
                v_id=v_id,
                initial_kf_name=kf_name,
                score=final_score,
                siglip_score=siglip_score,
                matched_objs=matched_objs,
                global_map=global_map,
                metadata=metadata,
                global_objects=global_objects,
                base_kf_dir=base_kf_dir,
                matched_transcript=matched_ts
            )
            display(card_widget)
        else:
            frame_idx = cand["frame_idx"]
            pts_time = int(cand["pts_time"])
            meta = metadata.get(v_id, {})
            yt_url = meta.get("watch_url", "")
            timestamp_link = f"{yt_url}&t={pts_time}s" if yt_url else "#"
            ts_info = f" | 🎙️ Transcript: {matched_ts['text']}" if matched_ts else ""
            print(f"Top {i+1}: {v_id}/{kf_name} | Score: {final_score:.4f} (SigLIP: {siglip_score:.4f}) | Frame: {frame_idx} | Time: {pts_time}s{ts_info} | {timestamp_link}")


