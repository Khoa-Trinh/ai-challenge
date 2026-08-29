import os
import json
import numpy as np
import faiss


def load_dataset_and_metadata(data_compile_dir):
    """
    Nạp mảng SigLIP Feature, Manifest, Global Map, và Processed Metadata từ data_compile_dir.
    """
    print("Đang nạp SigLIP Embeddings và Metadata...")
    features_path = os.path.join(data_compile_dir, "siglip_features.npy")
    features = np.load(features_path).astype(np.float32)

    with open(os.path.join(data_compile_dir, "manifest_keyframes.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(os.path.join(data_compile_dir, "global_map_keyframes.json"), "r", encoding="utf-8") as f:
        global_map = json.load(f)

    with open(os.path.join(data_compile_dir, "processed_metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"-> Đã nạp {len(manifest):,} keyframes vector | Shape: {features.shape}")
    return features, manifest, global_map, metadata


def build_faiss_index(features):
    """
    Tạo FAISS Cosine Index (IndexFlatIP).
    """
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    print("-> FAISS Index sẵn sàng!")
    return index
