import os
import glob
import json
import time
from tqdm import tqdm
import torch
from ultralytics import YOLO

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & THAM SỐ
# ==========================================
BASE_KF_DIR = "/kaggle/input/datasets/nguynhuyds/aic-dataset"
OUTPUT_DIR = "/kaggle/working"
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "global_objects.json")
CHECKPOINT_INTERVAL = 10000
BATCH_SIZE = 32
CONF_THRESHOLD = 0.20
MODEL_NAME = "yolov8x-worldv2.pt"  # Model YOLO-World v2-X chính xác nhất

# Danh sách từ vựng phong phú (Open-Vocabulary classes) mở rộng cho bối cảnh AIC & Việt Nam
EXTENDED_CLASSES = [
    # Phương tiện giao thông & Đường phố
    "person", "man", "woman", "child", "crowd", "police", "soldier", "doctor", "chef", "driver",
    "car", "motorcycle", "bicycle", "bus", "truck", "van", "ambulance", "fire truck", "taxi",
    "electric bike", "boat", "ship", "airplane", "helicopter", "train",
    "traffic light", "stop sign", "street sign", "billboard", "license plate", "helmet", "crosswalk",
    
    # Nhà bếp, Nấu ăn & Thực phẩm
    "knife", "cleaver", "cutting board", "pan", "pot", "stove", "gas cooker", "oven", "microwave",
    "bowl", "plate", "dish", "chopsticks", "spoon", "fork", "cup", "glass", "bottle", "teapot",
    "fish", "raw meat", "beef", "pork", "chicken", "shrimp", "crab", "vegetable", "fruit",
    "onion", "garlic", "chili", "lime", "lemon", "carrot", "tomato", "cucumber", "cabbage",
    "noodle", "rice", "soup", "bread", "cake", "sauce bottle", "cooking oil", "apron", "gloves",
    
    # Đồ gia dụng, Điện tử & Văn phòng
    "table", "desk", "chair", "sofa", "couch", "bed", "cabinet", "shelf", "refrigerator", "fan", "air conditioner",
    "television", "screen", "monitor", "laptop", "computer", "keyboard", "mouse", "cell phone", "smartphone",
    "camera", "microphone", "headphone", "speaker", "remote control", "book", "notebook", "pen", "paper", "document",
    "clock", "watch", "glasses", "sunglasses", "backpack", "handbag", "suitcase", "box", "package",
    
    # Trang phục & Phụ kiện
    "shirt", "t-shirt", "jacket", "coat", "suit", "dress", "skirt", "pants", "jeans", "shorts",
    "shoes", "sneakers", "boots", "sandals", "hat", "cap", "mask", "face mask", "tie", "belt",
    
    # Động vật & Tự nhiên
    "dog", "cat", "bird", "horse", "cow", "pig", "sheep", "elephant", "bear", "tiger", "lion", "monkey",
    "tree", "flower", "potted plant", "grass", "river", "lake", "bridge", "building", "house", "door", "window"
]


def collect_all_keyframes(base_dir):
    """
    Thu thập toàn bộ đường dẫn keyframe ảnh trong thư mục dataset.
    """
    print(f"🔍 Đang quét danh sách keyframe từ: {base_dir}")
    pattern = os.path.join(base_dir, "Keyframes_*", "keyframes", "*", "*.jpg")
    img_paths = glob.glob(pattern)

    # Nếu không tìm thấy dạng trên, thử tìm dạng phẳng
    if not img_paths:
        pattern_fallback = os.path.join(base_dir, "**", "*.jpg")
        img_paths = glob.glob(pattern_fallback, recursive=True)

    print(f"-> Tổng cộng tìm thấy: {len(img_paths):,} ảnh keyframes.")
    return sorted(img_paths)


def extract_key_from_path(img_path):
    """
    Chuyển đổi đường dẫn ảnh thành key chuẩn: 'L26_V411/094.jpg'
    """
    parts = os.path.normpath(img_path).split(os.sep)
    return f"{parts[-2]}/{parts[-1]}"


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚀 Bắt đầu trích xuất Objects với YOLO-World trên Device: {device}")

    # 1. Nạp Model YOLO-World
    model = YOLO(MODEL_NAME)
    print(f"-> Đang nạp {len(EXTENDED_CLASSES)} Open-Vocabulary classes...")
    model.set_classes(EXTENDED_CLASSES)

    # 2. Thu thập ảnh
    img_paths = collect_all_keyframes(BASE_KF_DIR)
    if not img_paths:
        print("❌ Không tìm thấy ảnh nào. Vui lòng kiểm tra lại BASE_KF_DIR!")
        return

    # 3. Tải kết quả cũ nếu đang chạy dở (Resume)
    global_objects = {}
    if os.path.exists(OUTPUT_JSON_PATH):
        print("📥 Tìm thấy checkpoint cũ, đang nạp dữ liệu để chạy tiếp...")
        with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
            global_objects = json.load(f)
        print(f"-> Đã xử lý trước đó: {len(global_objects):,} ảnh.")

    # Lọc các ảnh chưa xử lý
    pending_imgs = [p for p in img_paths if extract_key_from_path(p) not in global_objects]
    print(f"-> Còn lại: {len(pending_imgs):,} ảnh cần trích xuất.")

    if not pending_imgs:
        print("✅ Tất cả ảnh đã được trích xuất hoàn tất!")
        return

    # 4. Batch Inference
    start_time = time.time()
    total_batches = (len(pending_imgs) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(total_batches), desc="Extracting Objects"):
        batch_paths = pending_imgs[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]

        # Chạy inference hàng loạt
        results = model.predict(
            source=batch_paths,
            conf=CONF_THRESHOLD,
            device=device,
            verbose=False,
            half=True if torch.cuda.is_available() else False
        )

        for img_p, r in zip(batch_paths, results):
            key = extract_key_from_path(img_p)
            detected_classes = []

            if r.boxes is not None and len(r.boxes) > 0:
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                names_dict = r.names
                # Lấy danh sách unique các nhãn phát hiện được
                detected_classes = list(set([names_dict[cid] for cid in class_ids if cid in names_dict]))

            global_objects[key] = detected_classes

        # Lưu checkpoint định kỳ
        if (batch_idx + 1) % (CHECKPOINT_INTERVAL // BATCH_SIZE) == 0:
            with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(global_objects, f, ensure_ascii=False)
            tqdm.write(f"💾 Đã lưu Checkpoint: {len(global_objects):,} ảnh.")

    # 5. Lưu kết quả cuối cùng
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(global_objects, f, ensure_ascii=False, indent=2)

    elapsed = (time.time() - start_time) / 60
    print(f"\n🎉 HOÀN TẤT! Đã xuất: {OUTPUT_JSON_PATH}")
    print(f"⏱️ Tổng thời gian chạy: {elapsed:.2f} phút.")


if __name__ == "__main__":
    main()
