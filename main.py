import argparse
import yaml
import os

from dataset import load_dataset_and_metadata, build_faiss_index, build_bm25_index
from model import load_model, get_device
from retrieval import retrieve_kis, save_csv
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
        self.siglip_dir = config.get("siglip_dir", "/kaggle/input/datasets/nguynhuyds/siglip-feature")
        self.base_kf_dir = config.get("base_kf_dir", "/kaggle/input/datasets/nguynhuyds/aic-dataset")
        self.output_dir = config.get("output_dir", "/kaggle/working/submissions")
        self.device = get_device(config.get("device", "cuda"))

        self.retrieval_cfg = config.get("retrieval", {})
        self.viz_cfg = config.get("visualization", {})

        print(f"Khởi tạo AIC Pipeline trên Device: {self.device}")

        # 1. Model & Processor
        self.processor, self.model = load_model(self.model_name, self.device)

        # 2. Features & Metadata
        self.features, self.manifest, self.global_map, self.metadata = load_dataset_and_metadata(self.siglip_dir)

        # 3. FAISS Index
        self.index = build_faiss_index(self.features)

        # 4. BM25 Model
        self.video_ids, self.bm25_model = build_bm25_index(self.metadata)

    def retrieve(self, query_en_list, metadata_keywords="", top_k=None, spread_window=None, query_id=None, output_dir=None):
        top_k = top_k or self.retrieval_cfg.get("top_k", 100)
        spread_window = spread_window if spread_window is not None else self.retrieval_cfg.get("spread_window", 2)
        vector_search_top_k = self.retrieval_cfg.get("vector_search_top_k", 500)
        bm25_boost_weight = self.retrieval_cfg.get("bm25_boost_weight", 0.20)

        results = retrieve_kis(
            processor=self.processor,
            model=self.model,
            index=self.index,
            manifest=self.manifest,
            global_map=self.global_map,
            metadata=self.metadata,
            video_ids=self.video_ids,
            bm25_model=self.bm25_model,
            query_en_list=query_en_list,
            metadata_keywords=metadata_keywords,
            top_k=top_k,
            spread_window=spread_window,
            vector_search_top_k=vector_search_top_k,
            bm25_boost_weight=bm25_boost_weight,
            device=self.device,
        )

        if query_id:
            out_dir = output_dir or self.output_dir
            save_csv(results, query_id, output_dir=out_dir)

        return results

    def inspect(self, query_vi="", query_en_list=None, metadata_keywords="", top_n=None):
        if query_en_list is None:
            query_en_list = []
        top_n = top_n or self.viz_cfg.get("top_n", 6)
        max_length = self.viz_cfg.get("max_length", 64)
        vector_search_top_k = self.viz_cfg.get("vector_search_top_k", 200)
        bm25_boost_weight = self.viz_cfg.get("bm25_boost_weight", 0.20)

        inspect_query(
            processor=self.processor,
            model=self.model,
            index=self.index,
            manifest=self.manifest,
            global_map=self.global_map,
            metadata=self.metadata,
            video_ids=self.video_ids,
            bm25_model=self.bm25_model,
            query_vi=query_vi,
            query_en_list=query_en_list,
            metadata_keywords=metadata_keywords,
            top_n=top_n,
            base_kf_dir=self.base_kf_dir,
            vector_search_top_k=vector_search_top_k,
            bm25_boost_weight=bm25_boost_weight,
            max_length=max_length,
            device=self.device,
        )


def main():
    parser = argparse.ArgumentParser(description="AIC Video Retrieval Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", type=str, choices=["inspect", "retrieve"], default="inspect")
    parser.add_argument("--query_vi", type=str, default="", help="Vietnamese query description")
    parser.add_argument("--query_en", nargs="+", default=[], help="English query or sub-queries list")
    parser.add_argument("--metadata_keywords", type=str, default="", help="Metadata search keywords")
    parser.add_argument("--top_k", type=int, default=None, help="Top K results to retrieve")
    parser.add_argument("--top_n", type=int, default=None, help="Top N results to inspect visually")
    parser.add_argument("--spread_window", type=int, default=None, help="Frame spreading window")
    parser.add_argument("--query_id", type=str, default=None, help="Query ID for saving CSV")
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder for submissions")
    args = parser.parse_args()

    pipeline = AICPipeline(config_path=args.config)

    if not args.query_en:
        user_input = input("Nhập query tiếng Anh: ").strip()
        if user_input:
            args.query_en = [user_input]

    if args.mode == "inspect":
        pipeline.inspect(
            query_vi=args.query_vi,
            query_en_list=args.query_en,
            metadata_keywords=args.metadata_keywords,
            top_n=args.top_n,
        )
    elif args.mode == "retrieve":
        pipeline.retrieve(
            query_en_list=args.query_en,
            metadata_keywords=args.metadata_keywords,
            top_k=args.top_k,
            spread_window=args.spread_window,
            query_id=args.query_id or "query_submission",
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
