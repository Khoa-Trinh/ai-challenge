import os
import re
import base64
import shutil


def normalize_kf_name(kf_str):
    """
    Chuẩn hóa tên keyframe thành dạng xxx.jpg (ví dụ: '94' -> '094.jpg', '094.jpg' -> '094.jpg')
    """
    kf_str = kf_str.strip()
    if not kf_str:
        return ""
    if kf_str.endswith(".jpg"):
        name_part = kf_str[:-4]
        try:
            return f"{int(name_part):03d}.jpg"
        except ValueError:
            return kf_str
    else:
        try:
            return f"{int(kf_str):03d}.jpg"
        except ValueError:
            return f"{kf_str}.jpg"


def get_frame_idx(global_map, video_id, kf_name):
    """
    Tra cứu frame_idx từ global_map_keyframes.json
    """
    norm_kf = normalize_kf_name(kf_name)
    key = f"{video_id}/{norm_kf}"

    if key in global_map:
        info = global_map[key]
        if isinstance(info, dict):
            return info.get("frame_idx", 0)
        return int(info)

    # Thử tìm kiếm lùi nếu tên video hoặc kf có sai sót nhẹ
    for k, v in global_map.items():
        if k.lower() == key.lower():
            if isinstance(v, dict):
                return v.get("frame_idx", 0)
            return int(v)

    print(f"⚠️ Cảnh báo: Không tìm thấy keyframe '{key}' trong global_map. Dùng tạm frame_idx = 0.")
    return 0


def detect_query_type(filename):
    """
    Nhận diện loại truy vấn từ hậu tố file name: kis, qa, trake
    """
    name_clean = os.path.splitext(filename)[0].lower().strip()
    if name_clean.endswith("kis") or "_kis" in name_clean:
        return "kis"
    elif name_clean.endswith("qa") or "_qa" in name_clean:
        return "qa"
    elif name_clean.endswith("trake") or "_trake" in name_clean:
        return "trake"
    return None


def export_submission_interactive(global_map, output_dir="/kaggle/working/submissions"):
    """
    Quy trình tương tác từng bước tạo file CSV nộp bài:
    1. Nhập tên file (Hậu tố 'kis', 'qa', 'trake')
    2. Nhập danh sách Video ID (phân cách bằng khoảng trắng)
    3. Nhập danh sách Keyframes (xxx.jpg) cho từng Video ID
    4. Tra cứu global_map chuyển thành Frame ID và xuất file CSV + link tải trực tiếp (Base64 Data URI).
    """
    print("=" * 65)
    print("📋 AIC CSV EXPORT & SUBMISSION TOOL")
    print("=" * 65)

    # 1. Nhập tên file
    raw_filename = input("1️⃣ Nhập tên file truy vấn (vd: query_01_kis, query_02_qa, task1_trake): ").strip()
    if not raw_filename:
        print("❌ Tên file không được để trống!")
        return

    query_type = detect_query_type(raw_filename)
    if not query_type:
        print("\n❓ Không tự động nhận diện được loại truy vấn từ tên file.")
        print("  [1] Textual KIS (Format: <Video>, <Frame Idx>)")
        print("  [2] Q&A         (Format: <Video>, <Frame Idx>, \"<Answer>\")")
        print("  [3] TRAKE       (Format: <Video>, <Frame_1>, <Frame_2>, ..., <Frame_N>)")
        choice = input("👉 Chọn loại truy vấn (1, 2, 3) [Mặc định: 1]: ").strip() or "1"
        query_type = {"1": "kis", "2": "qa", "3": "trake"}.get(choice, "kis")

    clean_basename = os.path.splitext(raw_filename)[0]
    final_filename = f"{clean_basename}.csv"

    print(f"\n🎯 Loại truy vấn: [{query_type.upper()}] | Tên file xuất: {final_filename}")

    # 2. Nhập danh sách Video ID
    raw_vids = input("\n2️⃣ Nhập danh sách Video ID (phân cách bằng khoảng trắng, vd: L26_V411 L26_V360): ").strip()
    if not raw_vids:
        print("❌ Danh sách Video ID rỗng!")
        return

    video_ids = [v.strip() for v in raw_vids.split() if v.strip()]
    print(f"-> Đã nhận diện {len(video_ids)} Video: {video_ids}")

    # 3. Lặp qua từng Video ID để lấy Keyframes / Answers
    csv_rows = []

    print("\n3️⃣ Nhập thông tin keyframes cho từng Video:")
    for idx, v_id in enumerate(video_ids):
        print(f"\n--- [{idx + 1}/{len(video_ids)}] Video: {v_id} ---")

        if query_type == "kis":
            kf_input = input(f"   👉 Nhập keyframe(s) cho {v_id} (vd: 094.jpg 080.jpg hoặc 94 80): ").strip()
            kfs = [k.strip() for k in kf_input.split() if k.strip()]
            for kf in kfs:
                f_idx = get_frame_idx(global_map, v_id, kf)
                csv_rows.append(f"{v_id}, {f_idx}")
                print(f"      + {v_id}, {normalize_kf_name(kf)} -> Frame ID: {f_idx}")

        elif query_type == "qa":
            kf_input = input(f"   👉 Nhập keyframe cho {v_id} (vd: 094.jpg hoặc 94): ").strip()
            answer_input = input(f"   👉 Nhập đáp án Answer cho {v_id} (tối đa 100 ký tự): ").strip()
            if len(answer_input) > 100:
                print("   ⚠️ Đáp án dài hơn 100 ký tự, tự động cắt ngắn.")
                answer_input = answer_input[:100]

            f_idx = get_frame_idx(global_map, v_id, kf_input)
            csv_rows.append(f'{v_id}, {f_idx}, "{answer_input}"')
            print(f'      + {v_id}, Frame: {f_idx}, Answer: "{answer_input}"')

        elif query_type == "trake":
            kf_input = input(f"   👉 Nhập chuỗi sự kiện keyframes cho {v_id} theo thứ tự thời gian (vd: 012.jpg 045.jpg 089.jpg): ").strip()
            kfs = [k.strip() for k in kf_input.split() if k.strip()]
            frame_indices = [str(get_frame_idx(global_map, v_id, kf)) for kf in kfs]
            row_str = f"{v_id}, " + ", ".join(frame_indices)
            csv_rows.append(row_str)
            print(f"      + {row_str}")

    # Giới hạn tối đa 100 dòng theo quy chế
    if len(csv_rows) > 100:
        print(f"⚠️ Cảnh báo: Số lượng kết quả ({len(csv_rows)}) vượt quá giới hạn 100 dòng. Đã cắt về top 100 dòng.")
        csv_rows = csv_rows[:100]

    # 4. Xuất file CSV
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, final_filename)
    csv_content = "\n".join(csv_rows) + "\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    # Copy thêm ra /kaggle/working/ để hiện trực tiếp trong Output sidebar
    try:
        shutil.copy(out_path, f"/kaggle/working/{final_filename}")
    except Exception:
        pass

    print("\n" + "=" * 65)
    print(f"✅ ĐÃ XUẤT FILE THÀNH CÔNG: {out_path} ({len(csv_rows)} dòng)")
    print("=" * 65)

    # In preview nội dung file
    print("📄 Nội dung file CSV:")
    for r in csv_rows[:10]:
        print(f"   {r}")
    if len(csv_rows) > 10:
        print(f"   ... và {len(csv_rows) - 10} dòng khác.")

    # Mã hóa Base64 Data URI để tải trực tiếp từ trình duyệt KHÔNG CẦN QUA SERVER KAGGLE PROXY
    b64_csv = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
    data_uri = f"data:text/csv;charset=utf-8;base64,{b64_csv}"

    try:
        from IPython.display import display, HTML
        display(HTML(f"""
        <div style="margin-top: 15px; padding: 18px 24px; background: #064e3b; border-radius: 10px; border: 2px solid #10b981; max-width: 600px;">
            <b style="color: #6ee7b7; font-size: 17px;">📥 Tải file CSV nộp bài:</b><br>
            <p style="color: #cbd5e1; font-size: 14px; margin: 6px 0 14px 0;">Bấm nút bên dưới để tải trực tiếp file về máy tính:</p>
            <a href="{data_uri}" download="{final_filename}" 
               style="display: inline-block; background: #10b981; color: #000000; padding: 10px 24px; border-radius: 8px; font-weight: bold; font-size: 15px; text-decoration: none; box-shadow: 0 4px 14px rgba(16,185,129,0.4);">
                ⬇️ Download {final_filename}
            </a>
        </div>
        """))
    except Exception:
        pass

    return out_path
