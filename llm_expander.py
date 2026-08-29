import os
import json
import urllib.request
import urllib.error


SYSTEM_PROMPT = """You are an AI Video Retrieval expert for the AI Challenge competition (Text-to-Video Search using SigLIP).
Your task is to analyze a Vietnamese video query and extract visual search descriptions.

Instructions:
1. Ignore QA/trivia questions at the end (e.g., 'Hỏi đây là loài cá gì?', 'Quán trọ nằm trên đường nào?', 'Mỗi lần làm được bao nhiêu bánh?', 'Con số viết trên xe là gì?').
2. Classify into one of 2 modes:
   - "synonyms": The query describes a single scene, moment, or object with multiple attributes.
   - "sequential": The query describes 2 or more chronological events / storyline (e.g. 'Đầu tiên... sau đó...', 'góc quay bên trong... sau đó chuyển ra ngoài...').
3. For each step, create 2-3 rich, concrete English visual descriptions/synonyms. Highlight visual actions, objects, viewpoints (e.g., 'close up', 'cockpit view', 'exterior shot'), and settings.

Return ONLY a valid JSON object in the following format with no markdown formatting:
{
  "mode": "synonyms" or "sequential",
  "steps": [
    ["English visual synonym 1", "English visual synonym 2", "English visual synonym 3"]
  ]
}
"""

# Thứ tự ưu tiên model chính thức: Ưu tiên Flash 3.5 -> Flash 3.5 Lite (500 RPD) -> Flash 3.1 Lite (500 RPD) -> Flash 2.5 / 1.5
DEFAULT_MODEL_FALLBACKS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]


def expand_query_with_gemini(query_vi, api_key=None, model_candidates=None):
    """
    Tự động phân tích prompt Tiếng Việt với cơ chế Auto-Failover:
    Ưu tiên Gemini 3.5 Flash -> tự động nhảy sang Gemini 3.5 Flash Lite (500 RPD) hoặc Gemini 3.1 Flash Lite (500 RPD) nếu hết quota!
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Chưa cung cấp GEMINI_API_KEY! Hãy truyền api_key hoặc đặt biến môi trường.")

    models_to_try = model_candidates or DEFAULT_MODEL_FALLBACKS
    if isinstance(models_to_try, str):
        models_to_try = [models_to_try] + [m for m in DEFAULT_MODEL_FALLBACKS if m != models_to_try]

    last_error = None

    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{SYSTEM_PROMPT}\n\nVietnamese Query: {query_vi}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                content_text = resp_data["candidates"][0]["content"]["parts"][0]["text"]
                result_json = json.loads(content_text)
                print(f"✨ Phân tích thành công bằng Model: [{model_name}]")
                return result_json

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            last_error = f"HTTP {e.code} ({model_name}): {error_body}"
            if e.code in [429, 404, 400, 503]:
                print(f"⚠️ Model [{model_name}] (HTTP {e.code}) hết quota hoặc bận. Đang tự động chuyển sang model tiếp theo...")
                continue
            else:
                raise RuntimeError(f"Gemini API Error ({e.code}): {error_body}")
        except Exception as e:
            last_error = str(e)
            print(f"⚠️ Lỗi kết nối model [{model_name}]: {e}. Đang thử model tiếp theo...")
            continue

    raise RuntimeError(f"Tất cả các model Gemini đều thất bại. Lỗi cuối: {last_error}")
