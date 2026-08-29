import torch
from transformers import AutoProcessor, SiglipModel, AutoTokenizer, AutoModelForSeq2SeqLM


def get_device(device_name="cuda"):
    """
    Trả về device khả dụng (cuda nếu có, ngược lại cpu).
    """
    if device_name == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(model_name="google/siglip-so400m-patch14-384", device="cuda"):
    """
    Nạp Model SigLIP để encode text.
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()
    return processor, model


def encode_text_queries(processor, model, query_en_list, device="cuda", max_length=None, truncation=False):
    """
    Encode danh sách Text Sub-queries (đồng nghĩa / đa góc nhìn) và trả về vector trung bình chuẩn hóa.
    """
    if isinstance(query_en_list, str):
        query_en_list = [query_en_list]
    query_en_list = [str(q).strip() for q in query_en_list if str(q).strip()]

    if not query_en_list:
        raise ValueError("query_en_list không được để trống!")

    kwargs = {"padding": "max_length", "return_tensors": "pt"}
    if max_length is not None:
        kwargs["max_length"] = max_length
    if truncation:
        kwargs["truncation"] = truncation

    text_inputs = processor(text=query_en_list, **kwargs).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        if hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
        elif hasattr(text_features, 'last_hidden_state'):
            text_features = text_features.last_hidden_state[:, 0]

        t_embeds = text_features / text_features.norm(dim=-1, keepdim=True)
        # Average vector của tất cả sub-queries
        avg_embed = t_embeds.mean(dim=0, keepdim=True)
        avg_embed = avg_embed / avg_embed.norm(dim=-1, keepdim=True)
        query_vec = avg_embed.cpu().to(torch.float32).numpy()

    return query_vec


def load_translator(model_name="Helsinki-NLP/opus-mt-vi-en", device="cuda"):
    """
    Nạp model dịch tiếng Việt sang tiếng Anh Offline nhẹ (~300MB).
    """
    print(f"Đang nạp Model Dịch Offline ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    print("-> Model Dịch sẵn sàng!")
    return tokenizer, model


def translate_vi_to_en(tokenizer, model, texts, device="cuda", max_length=128):
    """
    Dịch tự động chuỗi, danh sách chuỗi, hoặc danh sách lồng (nested list) từ Tiếng Việt sang Tiếng Anh.
    """
    if isinstance(texts, str):
        inputs = tokenizer([texts.strip()], return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, max_length=max_length)
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    # Nếu là danh sách lồng: [[bước 1a, bước 1b], [bước 2a, bước 2b]]
    if isinstance(texts, list) and texts and isinstance(texts[0], (list, tuple)):
        nested_results = []
        for sub_list in texts:
            translated_sub = translate_vi_to_en(tokenizer, model, sub_list, device=device, max_length=max_length)
            nested_results.append(translated_sub)
        return nested_results

    # Nếu là danh sách phẳng: [câu 1, câu 2, ...]
    text_list = [str(t).strip() for t in texts if str(t).strip()]
    if not text_list:
        return []

    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs, max_length=max_length)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
