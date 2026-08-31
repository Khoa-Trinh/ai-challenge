import os
import json
import numpy as np
import faiss


def load_dataset_and_metadata(data_compile_dir):
    """
    Nạp mảng SigLIP Feature, Manifest, Global Map, Processed Metadata, và Global Objects từ data_compile_dir.
    """
    print(f"Đang nạp SigLIP Embeddings và Metadata từ: {data_compile_dir}")

    features_path = os.path.join(data_compile_dir, "siglip_features.npy")
    manifest_path = os.path.join(data_compile_dir, "manifest_keyframes.json")
    global_map_path = os.path.join(data_compile_dir, "global_map_keyframes.json")
    objects_path = os.path.join(data_compile_dir, "global_objects.json")
    inverted_objects_path = os.path.join(data_compile_dir, "inverted_objects.json")
    transcripts_path = os.path.join(data_compile_dir, "video_transcripts.json")
    metadata_path = os.path.join(data_compile_dir, "processed_metadata.json")

    for path in [features_path, manifest_path, global_map_path]:
        if not os.path.exists(path):
            # Thử tìm trong thư mục cha hoặc dạng phẳng nếu đường dẫn hơi khác
            alt_path = os.path.join(data_compile_dir, os.path.basename(path))
            if not os.path.exists(alt_path):
                raise FileNotFoundError(
                    f"Không tìm thấy file bắt buộc: '{path}'.\n"
                    f"Hãy chắc chắn bạn đã Add Dataset vào Kaggle và kiểm tra đúng data_compile_dir."
                )

    features = np.load(features_path).astype(np.float32)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(global_map_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
        # Hỗ trợ cả định dạng mới có key 'keyframes_map' lẫn định dạng cũ
        if isinstance(raw_map, dict) and "keyframes_map" in raw_map:
            global_map = raw_map["keyframes_map"]
        else:
            global_map = raw_map

    # Nạp metadata / transcripts nếu có
    metadata = {}
    if os.path.exists(transcripts_path):
        print(f"-> Đang nạp Video Transcripts: {transcripts_path}")
        with open(transcripts_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"-> Đã nạp Transcripts cho {len(metadata):,} videos.")
    elif os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    # Nạp Objects Detections & Inverted Index
    global_objects = {}
    if os.path.exists(objects_path):
        print(f"-> Đang nạp Global Objects: {objects_path}")
        with open(objects_path, "r", encoding="utf-8") as f:
            raw_objects = json.load(f)
            # Chuẩn hóa về dạng dict {tag: conf} để tra cứu nhanh
            for kf_k, v in raw_objects.items():
                if isinstance(v, dict) and "tags" in v and "scores" in v:
                    global_objects[kf_k] = dict(zip(v["tags"], v["scores"]))
                elif isinstance(v, dict):
                    global_objects[kf_k] = v
                elif isinstance(v, list):
                    global_objects[kf_k] = {tag: 1.0 for tag in v}
        print(f"-> Đã nạp thông tin Objects cho {len(global_objects):,} keyframes.")
    else:
        print(f"ℹ️ Không tìm thấy '{objects_path}'. Hệ thống vẫn hoạt động ở chế độ Vector Search thuần.")

    print(f"-> Đã nạp {len(manifest):,} keyframes vector | Shape: {features.shape}")
    return features, manifest, global_map, metadata, global_objects


def build_faiss_index(features):
    """
    Tạo FAISS Cosine Index (IndexFlatIP).
    """
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    print("-> FAISS Index sẵn sàng!")
    return index

