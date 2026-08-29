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
    metadata_path = os.path.join(data_compile_dir, "processed_metadata.json")
    objects_path = os.path.join(data_compile_dir, "global_objects.json")

    for path in [features_path, manifest_path, global_map_path, metadata_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Không tìm thấy file: '{path}'.\n"
                f"Hãy chắc chắn bạn đã Add Dataset vào Kaggle và kiểm tra đúng đường dẫn data_compile_dir."
            )

    features = np.load(features_path).astype(np.float32)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(global_map_path, "r", encoding="utf-8") as f:
        global_map = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    global_objects = {}
    if os.path.exists(objects_path):
        print(f"-> Đang nạp Global Objects Detections: {objects_path}")
        with open(objects_path, "r", encoding="utf-8") as f:
            global_objects = json.load(f)
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

