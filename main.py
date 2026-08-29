import yaml
import os
import sys
import importlib

import model
import dataset
import visualize


def hot_reload():
    """
    Tải lại tức thì các module Python theo đúng thứ tự phụ thuộc (model -> dataset -> visualize -> main).
    Xử lý an toàn khi đổi tên hàm, thêm/xóa module mà KHÔNG CẦN Restart Kernel!
    """
    modules_order = ["model", "dataset", "visualize", "main"]
    for mod_name in modules_order:
        if mod_name in sys.modules:
            try:
                importlib.reload(sys.modules[mod_name])
            except Exception:
                del sys.modules[mod_name]
                __import__(mod_name)
    print("⚡ Hot-reload thành công! Tất cả module và hàm mới đã được cập nhật.")


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

    def __init__(self, config_path="config.yaml", reuse_cache=True, **kwargs):
        config = load_config(config_path)
        config.update({k: v for k, v in kwargs.items() if v is not None})

        self.model_name = config.get("model_name", "google/siglip-so400m-patch14-384")
        self.data_compile_dir = config.get(
            "data_compile_dir",
            config.get("siglip_dir", "/kaggle/input/datasets/trnhngkhoashineekuwu/aic-compile-data")
        )
        self.base_kf_dir = config.get("base_kf_dir", "/kaggle/input/datasets/nguynhuyds/aic-dataset")
        self.device = model.get_device(config.get("device", "cuda"))
        self.viz_cfg = config.get("visualization", {})

        print(f"Khởi tạo AIC Pipeline trên Device: {self.device}")
        print(f"-> Thư mục compile dataset: {self.data_compile_dir}")

        # 1. Model & Processor (Dùng lại nếu đã có trong RAM)
        if reuse_cache and AICPipeline._cached_model is not None and AICPipeline._cached_processor is not None:
            print("-> Tái sử dụng Model & Processor từ bộ nhớ RAM/GPU (không cần nạp lại).")
            self.processor = AICPipeline._cached_processor
            self.model = AICPipeline._cached_model
        else:
            self.processor, self.model = model.load_model(self.model_name, self.device)
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
            self.features, self.manifest, self.global_map, self.metadata = dataset.load_dataset_and_metadata(self.data_compile_dir)
            AICPipeline._cached_features = self.features
            AICPipeline._cached_manifest = self.manifest
            AICPipeline._cached_global_map = self.global_map
            AICPipeline._cached_metadata = self.metadata

        # 3. FAISS Index (Dùng lại nếu đã có trong RAM)
        if reuse_cache and AICPipeline._cached_index is not None:
            print("-> Tái sử dụng FAISS Index từ bộ nhớ RAM.")
            self.index = AICPipeline._cached_index
        else:
            self.index = dataset.build_faiss_index(self.features)
            AICPipeline._cached_index = self.index

    def reload(self):
        """
        Hot-reload mã nguồn mới nhất mà vẫn giữ nguyên Model & Index trong RAM.
        """
        hot_reload()
        if "main" in sys.modules:
            self.__class__ = sys.modules["main"].AICPipeline

    def inspect(self, query: str = None, top_n=None):
        """
        Tìm kiếm và hiển thị kết quả trực quan cho một câu query Tiếng Anh (string).
        Nếu không truyền query, sẽ tự động mở giao diện interactive input UI.
        """
        if query is None or not str(query).strip():
            return self.input()

        top_n = top_n or self.viz_cfg.get("top_n", 6)
        max_length = self.viz_cfg.get("max_length", 64)
        vector_search_top_k = self.viz_cfg.get("vector_search_top_k", 200)

        # Gọi qua module visualize (đảm bảo luôn là phiên bản mới nhất sau hot_reload)
        visualize_mod = sys.modules.get("visualize", visualize)
        visualize_mod.inspect_query(
            processor=self.processor,
            model_obj=self.model,
            index=self.index,
            manifest=self.manifest,
            global_map=self.global_map,
            metadata=self.metadata,
            query_en=str(query).strip(),
            top_n=top_n,
            base_kf_dir=self.base_kf_dir,
            vector_search_top_k=vector_search_top_k,
            max_length=max_length,
            device=self.device,
        )

    def input(self):
        """
        Mở giao diện UI tìm kiếm trực quan bằng Widgets (Interactive Search Bar + Slider + Search Button).
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output

            query_box = widgets.Text(
                value='',
                placeholder='Nhập query tiếng Anh (ví dụ: preparing and cooking fish on cutting board)...',
                description='🔍 Query:',
                layout=widgets.Layout(width='60%')
            )

            top_n_slider = widgets.IntSlider(
                value=self.viz_cfg.get("top_n", 6),
                min=1,
                max=20,
                step=1,
                description='Top N:',
                layout=widgets.Layout(width='25%')
            )

            search_btn = widgets.Button(
                description=' Search',
                button_style='primary',
                icon='search',
                layout=widgets.Layout(width='120px')
            )

            output_area = widgets.Output()

            def execute_search(_=None):
                q = query_box.value.strip()
                if not q:
                    with output_area:
                        clear_output()
                        print("⚠️ Vui lòng nhập query tiếng Anh vào ô tìm kiếm!")
                    return
                with output_area:
                    clear_output()
                    self.inspect(query=q, top_n=top_n_slider.value)

            search_btn.on_click(execute_search)
            query_box.on_submit(execute_search)

            search_panel = widgets.VBox([
                widgets.HBox([query_box, top_n_slider, search_btn], layout=widgets.Layout(align_items='center', margin='10px 0')),
                output_area
            ])
            display(search_panel)
            return search_panel

        except ImportError:
            # Fallback nếu không có ipywidgets
            q = input("Nhập query tiếng Anh: ").strip()
            if q:
                self.inspect(query=q)
