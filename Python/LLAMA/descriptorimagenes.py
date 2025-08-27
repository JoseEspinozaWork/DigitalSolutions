# test_llava_ok.py
import requests, base64, json

IMAGE_PATH = r"capturaC.png"   # pon la ruta correcta si no está junto al .py
OLLAMA_URL = "http://localhost:11434/api/generate"

# 1) convertimos la imagen a base64
with open(IMAGE_PATH, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

# 2) payload: pedimos stream desactivado
payload = {
    "model": "llava:7b",
    "prompt": "Describe esta imagen en detalle",
    "images": [img_b64],
    "stream": False
}

# 3) petición
r = requests.post(OLLAMA_URL, json=payload, timeout=300)
r.raise_for_status()

data = r.json()  # ¡ya es un único JSON!
print("\n--- Texto del modelo ---\n")
print(data.get("response", ""))

# (opcional) ver todo el JSON de Ollama
# print(json.dumps(data, indent=2, ensure_ascii=False))
