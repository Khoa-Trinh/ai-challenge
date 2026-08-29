import torch
from transformers import AutoProcessor, SiglipModel


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


def encode_text_query(processor, model, query_text: str, device="cuda", max_length=64, truncation=True):
    """
    Encode câu query Tiếng Anh (chuỗi string) thành vector đặc trưng chuẩn hóa.
    """
    query_text = str(query_text).strip()

    kwargs = {
        "text": [query_text],
        "padding": "max_length",
        "return_tensors": "pt"
    }
    if max_length is not None:
        kwargs["max_length"] = max_length
    if truncation:
        kwargs["truncation"] = truncation

    text_inputs = processor(**kwargs).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        if hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output
        elif hasattr(text_features, 'last_hidden_state'):
            text_features = text_features.last_hidden_state[:, 0]

        t_embed = text_features / text_features.norm(dim=-1, keepdim=True)
        query_vec = t_embed.cpu().to(torch.float32).numpy()

    return query_vec
