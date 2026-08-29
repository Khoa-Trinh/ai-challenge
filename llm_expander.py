import os
import json
import urllib.request
import urllib.error


SYSTEM_PROMPT = """You are an AI Video Retrieval expert for the AI Challenge competition (Text-to-Video Search using SigLIP).
Your task is to analyze a Vietnamese video query and extract visual search descriptions.

Instructions:
1. Ignore QA/trivia questions at the end (e.g., 'Hỏi đây là loài cá gì?', 'Quán trọ nằm trên đường nào?', 'Mỗi lần làm được bao nhiêu bánh?').
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


def expand_query_with_gemini(query_vi, api_key=None, model_name="gemini-1.5-flash"):
    """
    Sử dụng Gemini Flash (1,500 RPD miễn phí) để tự động phân tích prompt Tiếng Việt thành:
    - Mode: 'synonyms' hoặc 'sequential'
    - Danh sách 2-3 câu Tiếng Anh mô tả thị giác cho từng bước.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Chưa cung cấp GEMINI_API_KEY! Hãy truyền api_key hoặc đặt biến môi trường.")

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
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            content_text = resp_data["candidates"][0]["content"]["parts"][0]["text"]
            result_json = json.loads(content_text)
            return result_json
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        raise RuntimeError(f"Gemini API Error ({e.code}): {error_msg}")
    except Exception as e:
        raise RuntimeError(f"Lỗi khi gọi Gemini API: {e}")
