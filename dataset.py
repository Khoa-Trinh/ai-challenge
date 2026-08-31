import os
import json
import numpy as np
import faiss


def load_dataset_and_metadata(data_compile_dir):
    """
    Nạp mảng SigLIP Feature, Manifest, Global Map, Video Transcripts, Global Objects và Inverted Objects từ data_compile_dir.
    """
    print(f"Đang nạp SigLIP Embeddings và Metadata từ: {data_compile_dir}")

    features_path = os.path.join(data_compile_dir, "siglip_features.npy")
    manifest_path = os.path.join(data_compile_dir, "manifest_keyframes.json")
    global_map_path = os.path.join(data_compile_dir, "global_map_keyframes.json")
    objects_path = os.path.join(data_compile_dir, "global_objects.json")
    inverted_objects_path = os.path.join(data_compile_dir, "inverted_objects.json")
    transcripts_path = os.path.join(data_compile_dir, "video_transcripts.json")

    for path in [features_path, manifest_path, global_map_path]:
        if not os.path.exists(path):
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
        if isinstance(raw_map, dict) and "keyframes_map" in raw_map:
            global_map = raw_map["keyframes_map"]
        else:
            global_map = raw_map

    # 1. Nạp Video Transcripts
    transcripts = {}
    if os.path.exists(transcripts_path):
        print(f"-> Đang nạp Video Transcripts: {transcripts_path}")
        with open(transcripts_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)
        print(f"-> Đã nạp Transcripts cho {len(transcripts):,} videos.")
    else:
        print(f"ℹ️ Không tìm thấy '{transcripts_path}'.")

    # 2. Nạp Global Objects
    global_objects = {}
    if os.path.exists(objects_path):
        print(f"-> Đang nạp Global Objects: {objects_path}")
        with open(objects_path, "r", encoding="utf-8") as f:
            raw_objects = json.load(f)
            for kf_k, v in raw_objects.items():
                if isinstance(v, dict) and "tags" in v and "scores" in v:
                    global_objects[kf_k] = dict(zip(v["tags"], v["scores"]))
                elif isinstance(v, dict):
                    global_objects[kf_k] = v
                elif isinstance(v, list):
                    global_objects[kf_k] = {tag: 1.0 for tag in v}
        print(f"-> Đã nạp thông tin Objects cho {len(global_objects):,} keyframes.")

    # 3. Nạp Inverted Objects (Chỉ mục đảo)
    inverted_objects = {}
    if os.path.exists(inverted_objects_path):
        print(f"-> Đang nạp Inverted Objects Index: {inverted_objects_path}")
        with open(inverted_objects_path, "r", encoding="utf-8") as f:
            inverted_objects = json.load(f)
        print(f"-> Đã nạp Inverted Index cho {len(inverted_objects):,} nhãn vật thể.")

    print(f"-> Đã nạp {len(manifest):,} keyframes vector | Shape: {features.shape}")
    return features, manifest, global_map, transcripts, global_objects, inverted_objects


def build_faiss_index(features):
    """
    Tạo FAISS Cosine Index (IndexFlatIP).
    """
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    print("-> FAISS Index sẵn sàng!")
    return index

