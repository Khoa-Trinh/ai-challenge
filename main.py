import argparse
import yaml
import os
import sys
import importlib

import dataset
import model
import visualize
from dataset import load_dataset_and_metadata, build_faiss_index
from model import load_model, load_translator, translate_vi_to_en, get_device
from visualize import inspect_query


def hot_reload():
    """
    Tải lại tức thì các module Python (visualize, dataset, model, main)
    ngay trong phiên làm việc của Jupyter Notebook mà KHÔNG CẦN Restart Kernel!
    """
    for mod in [visualize, dataset, model]:
        importlib.reload(mod)
    print("⚡ Hot-reload thành công! Tất cả code mới đã được cập nhật.")


def load_config(config_path="config.yaml"):
    # Tự động tìm config.yaml theo đường dẫn tương đối hoặc theo thư mục chứa main.py
    resolved_path = config_path
    if not os.path.isabs(resolved_path) and not os.path.exists(resolved_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(base_dir, config_path)
        if os.path.exists(alt_path):
            resolved_path = alt_path

    if os.path.exists(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg if isinstance(cfg, dict) else {}
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy file config tại '{config_path}', đang dùng giá trị mặc định.")
    return {}


class AICPipeline:
    # Class-level cache để giữ weights và embeddings trong RAM/GPU không phải nạp lại
    _cached_processor = None
    _cached_model = None
    _cached_features = None
    _cached_manifest = None
    _cached_global_map = None
    _cached_metadata = None
    _cached_index = None
    _cached_translator_tok = None
    _cached_translator_mod = None

    def __init__(self, config_path="config.yaml", reuse_cache=True, load_translator_on_start=False, **kwargs):
        config = load_config(config_path)
        config.update({k: v for k, v in kwargs.items() if v is not None})

        self.model_name = config.get("model_name", "google/siglip-so400m-patch14-384")
        self.translator_model_name = config.get("translator_model", "Helsinki-NLP/opus-mt-vi-en")
        self.data_compile_dir = config.get(
            "data_compile_dir",
            config.get("siglip_dir", "/kaggle/input/datasets/trnhngkhoashineekuwu/aic-compile-data")
        )
        self.base_kf_dir = config.get("base_kf_dir", "/kaggle/input/datasets/nguynhuyds/aic-dataset")
        self.device = get_device(config.get("device", "cuda"))
        self.viz_cfg = config.get("visualization", {})

        print(f"Khởi tạo AIC Pipeline trên Device: {self.device}")
        print(f"-> Thư mục compile dataset: {self.data_compile_dir}")

        # 1. SigLIP Model & Processor (Dùng lại nếu đã có trong RAM)
        if reuse_cache and AICPipeline._cached_model is not None and AICPipeline._cached_processor is not None:
            print("-> Tái sử dụng SigLIP Model từ bộ nhớ RAM/GPU.")
            self.processor = AICPipeline._cached_processor
            self.model = AICPipeline._cached_model
        else:
            self.processor, self.model = load_model(self.model_name, self.device)
            AICPipeline._cached_processor = self.processor
            AICPipeline._cached_model = self.model

        # 2. Features & Metadata (Dùng lại nếu đã có trong RAM)
        if reuse_cache and AICPipeline._cached_features is not None:
            print("-> Tái sử dụng SigLIP Embeddings và Metadata từ bộ nhớ RAM.")
            self.features = AICPipeline._cached_features
            self.manifest = AICPipeline._cached_manifest
            self.global_map = AICPipeline._cached_global_map
            self.metadata = AICPipeline._cached_metadata
        else:
            self.features, self.manifest, self.global_map, self.metadata = load_dataset_and_metadata(self.data_compile_dir)
            AICPipeline._cached_features = self.features
            AICPipeline._cached_manifest = self.manifest
            AICPipeline._cached_global_map = self.global_map
            AICPipeline._cached_metadata = self.metadata

        # 3. FAISS Index (Dùng lại nếu đã có trong RAM)
        if reuse_cache and AICPipeline._cached_index is not None:
            print("-> Tái sử dụng FAISS Index từ bộ nhớ RAM.")
            self.index = AICPipeline._cached_index
        else:
            self.index = build_faiss_index(self.features)
            AICPipeline._cached_index = self.index

        # 4. Local Neural Translator (Lazy load khi cần hoặc load trước)
        self.translator_tok = None
        self.translator_mod = None
        if load_translator_on_start:
            self._ensure_translator()

    def _ensure_translator(self):
        """Khởi tạo mô hình dịch nếu chưa nạp."""
        if self.translator_tok is None or self.translator_mod is None:
            if AICPipeline._cached_translator_tok is not None and AICPipeline._cached_translator_mod is not None:
                self.translator_tok = AICPipeline._cached_translator_tok
                self.translator_mod = AICPipeline._cached_translator_mod
            else:
                self.translator_tok, self.translator_mod = load_translator(self.translator_model_name, self.device)
                AICPipeline._cached_translator_tok = self.translator_tok
                AICPipeline._cached_translator_mod = self.translator_mod

    def translate(self, vi_texts):
        """
        Dịch tự động Tiếng Việt sang Tiếng Anh bằng mô hình cục bộ.
        """
        self._ensure_translator()
        return translate_vi_to_en(self.translator_tok, self.translator_mod, vi_texts, device=self.device)

    def reload(self):
        """
        Hot-reload mã nguồn mới nhất mà vẫn giữ nguyên Model & Index trong RAM.
        """
        hot_reload()

    def inspect(self, query_vi="", query_en_list=None, top_n=None):
        """
        Tìm kiếm hình ảnh. Nếu chỉ nhập query_vi, hệ thống sẽ TỰ ĐỘNG DỊCH sang Tiếng Anh.
        """
        # Xử lý tự động dịch nếu người dùng chỉ nhập Tiếng Việt
        if (query_en_list is None or len(query_en_list) == 0) and query_vi:
            print(f"🔄 Đang tự động dịch prompt Tiếng Việt bằng Local Model...")
            # Hỗ trợ tách nhiều câu nếu phân cách bằng dấu phẩy
            if isinstance(query_vi, str):
                vi_splits = [q.strip() for q in query_vi.split(",") if q.strip()]
            else:
                vi_splits = query_vi

            query_en_list = self.translate(vi_splits)
            if isinstance(query_en_list, str):
                query_en_list = [query_en_list]

            print(f"✅ Bản dịch Tiếng Anh: {query_en_list}")

        if query_en_list is None:
            query_en_list = []

        top_n = top_n or self.viz_cfg.get("top_n", 6)
        max_length = self.viz_cfg.get("max_length", 64)
        vector_search_top_k = self.viz_cfg.get("vector_search_top_k", 200)

        # Sử dụng hàm inspect_query từ module visualize
        visualize.inspect_query(
            processor=self.processor,
            model=self.model,
            index=self.index,
            manifest=self.manifest,
            global_map=self.global_map,
            metadata=self.metadata,
            query_vi=query_vi if isinstance(query_vi, str) else ", ".join(query_vi),
            query_en_list=query_en_list,
            top_n=top_n,
            base_kf_dir=self.base_kf_dir,
            vector_search_top_k=vector_search_top_k,
            max_length=max_length,
            device=self.device,
        )


def main():
    parser = argparse.ArgumentParser(description="AIC Video Search & Inspection Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--query_vi", type=str, default="", help="Vietnamese query description")
    parser.add_argument("--query_en", nargs="+", default=[], help="English query or sub-queries list")
    parser.add_argument("--top_n", type=int, default=None, help="Top N results to inspect visually")
    args = parser.parse_args()

    pipeline = AICPipeline(config_path=args.config)

    if not args.query_en and not args.query_vi:
        user_input = input("Nhập query (Tiếng Việt hoặc Tiếng Anh): ").strip()
        if user_input:
            args.query_vi = user_input

    pipeline.inspect(
        query_vi=args.query_vi,
        query_en_list=args.query_en,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
