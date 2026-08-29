import os
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi


def load_dataset_and_metadata(siglip_dir):
    """
    Nạp mảng SigLIP Feature, Manifest, Global Map, và Processed Metadata.
    """
    print("Đang nạp SigLIP Embeddings và Metadata...")
    features_path = os.path.join(siglip_dir, "siglip_features.npy")
    features = np.load(features_path).astype(np.float32)

    with open(os.path.join(siglip_dir, "manifest_keyframes.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(os.path.join(siglip_dir, "global_map_keyframes.json"), "r", encoding="utf-8") as f:
        global_map = json.load(f)

    with open(os.path.join(siglip_dir, "processed_metadata.json"), "r", encoding="utf-8") as f:
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


def build_bm25_index(metadata):
    """
    Xây dựng BM25 cho Metadata Search.
    """
    video_ids = list(metadata.keys())
    bm25_corpus = [metadata[v]["text_corpus"].split() for v in video_ids]
    bm25_model = BM25Okapi(bm25_corpus)
    print(f"-> Đã index BM25 cho {len(video_ids)} videos metadata.")
    return video_ids, bm25_model
