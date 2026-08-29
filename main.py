import argparse
import yaml
import os

from dataset import load_dataset_and_metadata, build_faiss_index
from model import load_model, get_device
from visualize import inspect_query


def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class AICPipeline:
    def __init__(self, config_path="config.yaml", **kwargs):
        config = load_config(config_path)
        config.update({k: v for k, v in kwargs.items() if v is not None})

        self.model_name = config.get("model_name", "google/siglip-so400m-patch14-384")
        self.data_compile_dir = config.get(
            "data_compile_dir",
            config.get("siglip_dir", "/kaggle/input/datasets/trnhngkhoashineekuwu/aic-compile-data")
        )
        self.base_kf_dir = config.get("base_kf_dir", "/kaggle/input/datasets/nguynhuyds/aic-dataset")
        self.device = get_device(config.get("device", "cuda"))

        self.viz_cfg = config.get("visualization", {})

        print(f"Khởi tạo AIC Pipeline trên Device: {self.device}")

        # 1. Model & Processor
        self.processor, self.model = load_model(self.model_name, self.device)

        # 2. Features & Metadata
        self.features, self.manifest, self.global_map, self.metadata = load_dataset_and_metadata(self.data_compile_dir)

        # 3. FAISS Index
        self.index = build_faiss_index(self.features)

    def inspect(self, query_vi="", query_en_list=None, top_n=None):
        if query_en_list is None:
            query_en_list = []
        top_n = top_n or self.viz_cfg.get("top_n", 6)
        max_length = self.viz_cfg.get("max_length", 64)
        vector_search_top_k = self.viz_cfg.get("vector_search_top_k", 200)

        inspect_query(
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


def main():
    parser = argparse.ArgumentParser(description="AIC Video Search & Inspection Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--query_vi", type=str, default="", help="Vietnamese query description")
    parser.add_argument("--query_en", nargs="+", default=[], help="English query or sub-queries list")
    parser.add_argument("--top_n", type=int, default=None, help="Top N results to inspect visually")
    args = parser.parse_args()

    pipeline = AICPipeline(config_path=args.config)

    if not args.query_en:
        user_input = input("Nhập query tiếng Anh: ").strip()
        if user_input:
            args.query_en = [user_input]

    pipeline.inspect(
        query_vi=args.query_vi,
        query_en_list=args.query_en,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
