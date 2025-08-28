# describe_image_ollama.py
# Usa Ollama (local) para describir imágenes: juego/app/idle, software y código/lenguaje si aplica.
# Ejecuta: python describe_image_ollama.py ruta\a\imagen.png
# Opcional: set VLMODEL=llama3.2-vision  (por defecto usa "moondream")

import os
import sys
import json
import base64
import requests
from typing import Dict, Any, List

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
MODEL = os.environ.get("VLMODEL", "moondream")  # "moondream" o "llama3.2-vision"

PROMPT = (
    "Analiza la imagen. Devuelve SOLO JSON con estas claves exactas en minúscula:\n"
    "{actividad, app, lenguaje, descripcion}.\n"
    "- actividad ∈ {juego, aplicación, programación, escritorio inactivo}\n"
    "- app: nombre del programa/juego si lo reconoces; null si no\n"
    "- lenguaje: si hay código, di el lenguaje (Java, Python, C#, etc.); null si no\n"
    "- descripcion: breve resumen en español de lo que se ve"
)

def b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_ollama(image_paths: List[str]) -> str:
    """Llama al endpoint /api/generate de Ollama con 1+ imágenes y retorna el texto crudo."""
    images64 = [b64_image(p) for p in image_paths]
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": images64,
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()

def parse_json_or_wrap(text: str) -> Dict[str, Any]:
    """Intenta leer el JSON devuelto; si falla, lo envuelve en un JSON mínimo."""
    try:
        return json.loads(text)
    except Exception:
        # fallback: intenta extraer el primer bloque {...}
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {
            "actividad": None,
            "app": None,
            "lenguaje": None,
            "descripcion": None,
            "raw_text": text
        }

def describe_images(paths: List[str]) -> Dict[str, Any]:
    # Verifica archivos
    checked = []
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"No existe el archivo: {p}")
        checked.append(p)
    # Llama a Ollama
    text = call_ollama(checked)
    return parse_json_or_wrap(text)

def main():
    if len(sys.argv) < 2:
        print("Uso:\n  python describe_image_ollama.py imagen.png\n"
              "  (opcional) set VLMODEL=llama3.2-vision")
        sys.exit(1)
    paths = sys.argv[1:]
    try:
        result = describe_images(paths)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
