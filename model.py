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


def load_translator(model_name="Helsinki-NLP/opus-mt-vi-en", device="cuda"):
    """
    Nạp mô hình dịch thuật nhẹ, chính xác cục bộ (Helsinki-NLP/opus-mt-vi-en ~300MB).
    Chạy 100% offline trên GPU/CPU, không cần mạng hay Google Translate.
    """
    print(f"🌐 Đang nạp mô hình dịch Tiếng Việt -> Tiếng Anh ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    translator_model.eval()
    print("-> Mô hình dịch thuật sẵn sàng!")
    return tokenizer, translator_model


def translate_vi_to_en(tokenizer, translator_model, texts, device="cuda"):
    """
    Dịch chuỗi hoặc danh sách chuỗi Tiếng Việt sang Tiếng Anh.
    """
    if not texts:
        return "" if isinstance(texts, str) else []

    is_single = isinstance(texts, str)
    text_list = [texts] if is_single else texts
    cleaned_list = [str(t).strip() for t in text_list if str(t).strip()]

    if not cleaned_list:
        return "" if is_single else []

    inputs = tokenizer(cleaned_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        translated_tokens = translator_model.generate(**inputs, max_length=128, num_beams=4)
        translated_texts = [
            tokenizer.decode(t, skip_special_tokens=True).strip() for t in translated_tokens
        ]

    return translated_texts[0] if is_single else translated_texts


def encode_text_queries(processor, model, query_en_list, device="cuda", max_length=None, truncation=False):
    """
    Encode danh sách Text Sub-queries và trả về vector đặc trưng trung bình chuẩn hóa.
    """
    if isinstance(query_en_list, str):
        query_en_list = [query_en_list]
    query_en_list = [str(q).strip() for q in query_en_list if str(q).strip()]

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
