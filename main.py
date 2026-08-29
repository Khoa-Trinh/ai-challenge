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
from visualize import inspect_query, inspect_sequential_query


def hot_reload():
    """
    Tải lại tức thì các module Python (visualize, dataset, model)
    ngay trong phiên làm việc của Jupyter Notebook mà KHÔNG CẦN Restart Kernel!
    """
    for mod in [visualize, dataset, model]:
        importlib.reload(mod)
    print("⚡ Hot-reload thành công! Tất cả code mới đã được cập nhật.")


def load_config(config_path="config.yaml"):
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

    def __init__(self, config_path="config.yaml", reuse_cache=True, load_translation=False, **kwargs):
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

        # 1. Model & Processor
        if reuse_cache and AICPipeline._cached_model is not None and AICPipeline._cached_processor is not None:
            print("-> Tái sử dụng SigLIP Model & Processor từ bộ nhớ RAM/GPU.")
            self.processor = AICPipeline._cached_processor
            self.model = AICPipeline._cached_model
        else:
            self.processor, self.model = load_model(self.model_name, self.device)
            AICPipeline._cached_processor = self.processor
            AICPipeline._cached_model = self.model

        # 2. Features & Metadata
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

        # 3. FAISS Index
        if reuse_cache and AICPipeline._cached_index is not None:
            print("-> Tái sử dụng FAISS Index từ bộ nhớ RAM.")
            self.index = AICPipeline._cached_index
        else:
            self.index = build_faiss_index(self.features)
            AICPipeline._cached_index = self.index

        # 4. Offline Translator (nạp khi cần hoặc nếu load_translation=True)
        self.translator_tok = None
        self.translator_mod = None
        if load_translation:
            self._init_translator()

    def _init_translator(self):
        if AICPipeline._cached_translator_tok is not None and AICPipeline._cached_translator_mod is not None:
            self.translator_tok = AICPipeline._cached_translator_tok
            self.translator_mod = AICPipeline._cached_translator_mod
        else:
            self.translator_tok, self.translator_mod = load_translator(self.translator_model_name, self.device)
            AICPipeline._cached_translator_tok = self.translator_tok
            AICPipeline._cached_translator_mod = self.translator_mod

    def translate(self, vi_texts):
        """
        Dịch chuỗi hoặc danh sách chuỗi Tiếng Việt sang Tiếng Anh.
        """
        if self.translator_tok is None:
            self._init_translator()
        return translate_vi_to_en(self.translator_tok, self.translator_mod, vi_texts, device=self.device)

    def reload(self):
        """
        Hot-reload mã nguồn mới nhất mà vẫn giữ nguyên Model & Index trong RAM.
        """
        hot_reload()

    def inspect(self, query_vi="", query_en_list=None, top_n=None):
        """
        Mode 1: Synonyms / Single Scene Inspection
        """
        if query_en_list is None:
            query_en_list = []
        top_n = top_n or self.viz_cfg.get("top_n", 6)
        max_length = self.viz_cfg.get("max_length", 64)
        vector_search_top_k = self.viz_cfg.get("vector_search_top_k", 200)

        visualize.inspect_query(
            processor=self.processor,
            model=self.model,
            index=self.index,
            manifest=self.manifest,
            global_map=self.global_map,
            metadata=self.metadata,
            query_vi=query_vi,
            query_en_list=query_en_list,
            top_n=top_n,
            base_kf_dir=self.base_kf_dir,
            vector_search_top_k=vector_search_top_k,
            max_length=max_length,
            device=self.device,
        )

    def inspect_sequential(self, steps_en_list, steps_vi_list=None, top_n=None, max_time_gap=None):
        """
        Mode 2: Sequential Actions / Timeline Inspection
        """
        top_n = top_n or self.viz_cfg.get("top_n", 6)
        max_length = self.viz_cfg.get("max_length", 64)
        vector_search_top_k = self.viz_cfg.get("sequential_search_top_k", 300)
        max_time_gap = max_time_gap or self.viz_cfg.get("max_time_gap", 240)

        visualize.inspect_sequential_query(
            processor=self.processor,
            model=self.model,
            index=self.index,
            manifest=self.manifest,
            global_map=self.global_map,
            metadata=self.metadata,
            steps_en_list=steps_en_list,
            steps_vi_list=steps_vi_list,
            top_n=top_n,
            base_kf_dir=self.base_kf_dir,
            vector_search_top_k=vector_search_top_k,
            max_time_gap=max_time_gap,
            max_length=max_length,
            device=self.device,
        )

    def interactive(self):
        """
        Menu tương tác trực quan ngay trong Notebook để chọn Mode và nhập query.
        """
        print("="*60)
        print("🎯 AIC INTERACTIVE VIDEO SEARCH MENU")
        print("  [1] Synonyms Mode (Cùng 1 cảnh / Đa góc nhìn / Vector Mean)")
        print("  [2] Sequential Mode (Chuỗi hành động tuần tự theo thời gian)")
        print("="*60)
        mode = input("👉 Chọn Mode (1 hoặc 2) [Mặc định: 1]: ").strip() or "1"

        use_translate = input("🌐 Bạn có muốn nhập Tiếng Việt và tự động dịch sang Tiếng Anh? (y/n) [n]: ").strip().lower() == 'y'

        if mode == "2":
            print("\n--- 🎬 SEQUENTIAL MODE (Nhập từng bước hành động) ---")
            n_steps = int(input("Số bước hành động (vd: 2 hoặc 3): ").strip() or "2")
            steps_input = []
            for s in range(n_steps):
                step_text = input(f"  Nhập mô tả Bước {s+1}: ").strip()
                steps_input.append(step_text)

            if use_translate:
                print("🔄 Đang tự động dịch sang Tiếng Anh...")
                steps_en = self.translate(steps_input)
                print(f"-> Bản dịch: {steps_en}")
                self.inspect_sequential(steps_en_list=steps_en, steps_vi_list=steps_input)
            else:
                self.inspect_sequential(steps_en_list=steps_input, steps_vi_list=steps_input)

        else:
            print("\n--- 🔎 SYNONYMS MODE (Mô tả cùng 1 cảnh) ---")
            raw_input = input("Nhập câu query (hoặc các sub-query cách nhau bằng dấu phẩy ','): ").strip()
            queries = [q.strip() for q in raw_input.split(",") if q.strip()]

            if use_translate:
                print("🔄 Đang tự động dịch sang Tiếng Anh...")
                queries_en = self.translate(queries)
                print(f"-> Bản dịch: {queries_en}")
                self.inspect(query_vi=raw_input, query_en_list=queries_en)
            else:
                self.inspect(query_vi=raw_input, query_en_list=queries)


def main():
    parser = argparse.ArgumentParser(description="AIC Video Search & Inspection Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, choices=["synonyms", "sequential"], default="synonyms")
    parser.add_argument("--query_vi", type=str, default="", help="Vietnamese query description")
    parser.add_argument("--query_en", nargs="+", default=[], help="English query or sub-queries list")
    parser.add_argument("--translate", action="store_true", help="Auto translate Vietnamese queries")
    parser.add_argument("--top_n", type=int, default=None, help="Top N results to inspect visually")
    args = parser.parse_args()

    pipeline = AICPipeline(config_path=args.config, load_translation=args.translate)

    if not args.query_en and not args.query_vi:
        pipeline.interactive()
        return

    if args.translate and args.query_vi and not args.query_en:
        args.query_en = [pipeline.translate(args.query_vi)]

    if args.mode == "sequential":
        pipeline.inspect_sequential(
            steps_en_list=args.query_en,
            steps_vi_list=[args.query_vi] if args.query_vi else None,
            top_n=args.top_n,
        )
    else:
        pipeline.inspect(
            query_vi=args.query_vi,
            query_en_list=args.query_en,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
