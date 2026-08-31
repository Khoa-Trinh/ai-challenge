import yaml
import os
import sys
import importlib

import model
import dataset
import visualize
import export


def hot_reload():
    """
    Tải lại tức thì các module Python theo đúng thứ tự phụ thuộc (model -> dataset -> visualize -> export -> main).
    Xử lý an toàn khi đổi tên hàm, thêm/xóa module mà KHÔNG CẦN Restart Kernel!
    """
    modules_order = ["model", "dataset", "transcript", "visualize", "export", "main"]
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
    _cached_global_objects = None
    _cached_inverted_objects = None
    _cached_index = None

    def __init__(self, config_path="config.yaml", reuse_cache=True, **kwargs):
        config = load_config(config_path)
        config.update({k: v for k, v in kwargs.items() if v is not None})

        self.model_name = config.get("model_name", "google/siglip-so400m-patch14-384")
        self.data_compile_dir = config.get(
            "data_compile_dir",
            config.get("siglip_dir", "/kaggle/input/datasets/trnhngkhoashineekuwu/aic-compiled-data")
        )
        self.base_kf_dir = config.get("base_kf_dir", "/kaggle/input/datasets/nguynhuyds/aic-dataset")
        self.output_dir = config.get("output_dir", "/kaggle/working/submissions")
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

        # 2. Features & Metadata & Objects (Dùng lại nếu đã có trong RAM)
        if reuse_cache and AICPipeline._cached_features is not None:
            print("-> Tái sử dụng SigLIP Embeddings, Metadata và Objects từ bộ nhớ RAM.")
            self.features = AICPipeline._cached_features
            self.manifest = AICPipeline._cached_manifest
            self.global_map = AICPipeline._cached_global_map
            self.metadata = AICPipeline._cached_metadata
            self.global_objects = AICPipeline._cached_global_objects
            self.inverted_objects = AICPipeline._cached_inverted_objects
        else:
            self.features, self.manifest, self.global_map, self.metadata, self.global_objects, self.inverted_objects = dataset.load_dataset_and_metadata(self.data_compile_dir)
            AICPipeline._cached_features = self.features
            AICPipeline._cached_manifest = self.manifest
            AICPipeline._cached_global_map = self.global_map
            AICPipeline._cached_metadata = self.metadata
            AICPipeline._cached_global_objects = self.global_objects
            AICPipeline._cached_inverted_objects = self.inverted_objects

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

    def inspect(self, query: str = None, query_en: str = None, query_vn: str = None, top_n=None, use_objects=None, object_weight=None, use_inverted=None, use_transcripts=None, transcript_weight=None):
        """
        Tìm kiếm đa phương thức (SigLIP Vector + Objects qua query_en; Speech Transcripts qua query_vn).
        Nếu không truyền query, sẽ tự động mở giao diện interactive input UI.
        """
        q_en = query_en or query
        q_vn = query_vn

        if (q_en is None or not str(q_en).strip()) and (q_vn is None or not str(q_vn).strip()):
            self.input()
            return

        top_n = top_n or self.viz_cfg.get("top_n", 10)
        max_length = self.viz_cfg.get("max_length", 64)
        vector_search_top_k = self.viz_cfg.get("vector_search_top_k", 200)
        
        if use_objects is None:
            use_objects = self.viz_cfg.get("use_objects", False)
        if object_weight is None:
            object_weight = self.viz_cfg.get("object_weight", 0.15)
        if use_inverted is None:
            use_inverted = self.viz_cfg.get("use_inverted", False)
        if use_transcripts is None:
            use_transcripts = self.viz_cfg.get("use_transcripts", False)
        if transcript_weight is None:
            transcript_weight = self.viz_cfg.get("transcript_weight", 0.35)

        visualize_mod = sys.modules.get("visualize", visualize)
        visualize_mod.inspect_query(
            processor=self.processor,
            model_obj=self.model,
            index=self.index,
            manifest=self.manifest,
            global_map=self.global_map,
            metadata=self.metadata,
            query_en=str(q_en).strip() if q_en else "",
            query_vn=str(q_vn).strip() if q_vn else "",
            top_n=top_n,
            global_objects=self.global_objects if use_objects else None,
            inverted_objects=self.inverted_objects if use_inverted else None,
            use_objects=use_objects or use_inverted,
            object_weight=object_weight,
            use_transcripts=use_transcripts,
            transcript_weight=transcript_weight,
            base_kf_dir=self.base_kf_dir,
            vector_search_top_k=vector_search_top_k,
            max_length=max_length,
            device=self.device,
        )

    def input(self):
        """
        Mở giao diện UI tìm kiếm trực quan 3 dòng:
        - Row 1: English Query cho SigLIP & Objects
        - Row 2: Vietnamese Query cho Speech Transcripts
        - Row 3: Top N Slider, Toggles & Nút Tìm kiếm
        """
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output

            # DÒNG 1: Ô nhập Query tiếng Anh (SigLIP & Objects)
            en_query_box = widgets.Text(
                value='',
                placeholder='Mô tả tiếng Anh cho Hình ảnh & Objects (ví dụ: chef preparing and cooking fish on cutting board)...',
                description='🇬🇧 English:',
                layout=widgets.Layout(width='100%')
            )

            # DÒNG 2: Ô nhập Query tiếng Việt (Speech Transcripts)
            vn_query_box = widgets.Text(
                value='',
                placeholder='Từ khóa / Lời thoại tiếng Việt cho Transcripts (ví dụ: cực quang, đàn hổ con, COVID-19)...',
                description='🇻🇳 Tiếng Việt:',
                layout=widgets.Layout(width='100%')
            )

            # DÒNG 3: Các nút điều khiển & Toggles (Default False, Top N = 10)
            top_n_slider = widgets.IntSlider(
                value=self.viz_cfg.get("top_n", 10),
                min=1,
                max=50,
                step=1,
                description='Top N:',
                layout=widgets.Layout(width='200px')
            )

            chk_use_objects = widgets.Checkbox(
                value=self.viz_cfg.get("use_objects", False),
                description='🎯 Objects',
                indent=False,
                layout=widgets.Layout(width='100px')
            )

            chk_use_inverted = widgets.Checkbox(
                value=self.viz_cfg.get("use_inverted", False),
                description='🔄 Inverted',
                indent=False,
                layout=widgets.Layout(width='105px')
            )

            chk_use_transcripts = widgets.Checkbox(
                value=self.viz_cfg.get("use_transcripts", False),
                description='🎙️ Transcripts',
                indent=False,
                layout=widgets.Layout(width='125px')
            )

            search_btn = widgets.Button(
                description=' Tìm kiếm',
                button_style='primary',
                icon='search',
                layout=widgets.Layout(width='130px', height='36px')
            )

            output_area = widgets.Output()

            def on_search_clicked(b):
                with output_area:
                    clear_output(wait=True)
                    q_en = en_query_box.value.strip()
                    q_vn = vn_query_box.value.strip()
                    if not q_en and not q_vn:
                        print("⚠️ Vui lòng nhập ít nhất một câu mô tả tiếng Anh hoặc từ khóa tiếng Việt.")
                        return
                    self.inspect(
                        query_en=q_en,
                        query_vn=q_vn,
                        top_n=top_n_slider.value,
                        use_objects=chk_use_objects.value,
                        use_inverted=chk_use_inverted.value,
                        use_transcripts=chk_use_transcripts.value
                    )

            search_btn.on_click(on_search_clicked)
            en_query_box.on_submit(lambda _: on_search_clicked(None))
            vn_query_box.on_submit(lambda _: on_search_clicked(None))

            row1 = widgets.HBox([en_query_box], layout=widgets.Layout(width='100%', margin='0 0 8px 0'))
            row2 = widgets.HBox([vn_query_box], layout=widgets.Layout(width='100%', margin='0 0 10px 0'))
            row3 = widgets.HBox(
                [top_n_slider, chk_use_objects, chk_use_inverted, chk_use_transcripts, search_btn],
                layout=widgets.Layout(align_items='center', margin='0 0 15px 0')
            )

            ui = widgets.VBox([row1, row2, row3, output_area])
            display(ui)
            return None
        except ImportError:
            q = input("Nhập query: ").strip()
            if q:
                self.inspect(query=q)

    def export(self, output_dir=None):
        """
        Khởi chạy công cụ xuất kết quả nộp bài ra file CSV theo chuẩn quy định (KIS, Q&A, TRAKE)
        và cung cấp link tải trực tiếp.
        """
        out_dir = output_dir or self.output_dir
        export_mod = sys.modules.get("export", export)
        return export_mod.export_submission_interactive(
            global_map=self.global_map,
            output_dir=out_dir
        )

