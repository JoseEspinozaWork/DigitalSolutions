import os, sys, json, base64, requests, re

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
IMG_PATH_ARG = sys.argv[1] if len(sys.argv) > 1 else None
MODEL = "llava:7b"  # Cambia a 'llava:13b' si ya lo tienes descargado

def check_server():
    try:
        r = requests.get(f"{OLLAMA}/api/tags", timeout=10)
        r.raise_for_status()
        return True, r.json()
    except Exception as e:
        return False, str(e)

def read_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def pick_image_path():
    path = IMG_PATH_ARG
    if not path:
        print("Escribe la ruta completa de la imagen (PNG/JPG):")
        path = input("> ").strip('"').strip()
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")
    return path

# --- NUEVO: normalizador para 1–2 frases máximo ---
def enforce_brief(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""
    # 1) quitar saltos de línea y espacios extra
    t = re.sub(r"\s+", " ", text.strip())
    # 2) cortar por frases (., !, ?) manteniendo el separador
    parts = re.split(r"(?<=[.!?])\s+", t)
    # 3) tomar las primeras N frases no vacías
    parts = [p.strip() for p in parts if p.strip()]
    brief = " ".join(parts[:max_sentences])
    # seguridad: si no termina en puntuación, añade punto
    if brief and brief[-1] not in ".!?":
        brief += "."
    return brief

def call_generate(img_b64: str):
    url = f"{OLLAMA}/api/generate"
    payload = {
        "model": MODEL,
        "prompt": (
            "Responde SOLO en español y en una o dos frases como máximo, SIN saltos de línea. "
            "Di si la captura muestra código (con lenguaje y entorno si lo reconoces), un juego, "
            "una app de oficina, un navegador u otra app; y qué está haciendo el usuario. "
            "No añadas detalles largos ni listas."
        ),
        "images": [img_b64],
        "stream": False,
        # Si tu build da error con options en /api/generate, comenta este bloque:
        "options": {
            "temperature": 0.1,
            "num_predict": 256
        }
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()

def call_chat(img_b64: str):
    url = f"{OLLAMA}/api/chat"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": "Responde SIEMPRE en español, en 1–2 frases máximo y SIN saltos de línea."},
            {"role": "user",
             "content": ("En 1–2 frases: di si la captura muestra código (lenguaje y entorno si lo reconoces), "
                         "un juego, app de oficina, navegador u otra app; y qué está haciendo el usuario."),
             "images": [img_b64]}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 256
        }
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()

def main():
    print(f"[INFO] Verificando servidor Ollama en {OLLAMA} ...")
    ok, info = check_server()
    if not ok:
        print("[ERROR] No puedo conectar con Ollama.")
        print("Detalle:", info)
        sys.exit(1)
    else:
        print("[OK] Servidor activo.")

    try:
        img_path = pick_image_path()
    except Exception as e:
        print("[ERROR]", e); sys.exit(2)

    print(f"[INFO] Leyendo imagen: {img_path}")
    img_b64 = read_b64(img_path)

    # 1) Intento /api/generate
    try:
        print("[INFO] Llamando /api/generate ...")
        data = call_generate(img_b64)
        raw = (data.get("response") or "").strip()
        print("\n--- SALIDA ---\n")
        print(enforce_brief(raw))
        return
    except requests.HTTPError as he:
        txt = he.response.text if he.response is not None else str(he)
        print("[WARN] /api/generate falló. Intentaré /api/chat.")
        print("Detalles:", txt[:500])
    except Exception as e:
        print("[WARN] /api/generate falló (excepción). Intentaré /api/chat.")
        print("Detalles:", str(e))

    # 2) Fallback /api/chat
    try:
        print("[INFO] Llamando /api/chat ...")
        data = call_chat(img_b64)
        raw = (data.get("message") or {}).get("content", "")
        print("\n--- SALIDA ---\n")
        print(enforce_brief(raw))
    except Exception as e:
        print("[ERROR] También falló /api/chat.")
        print("Detalles:", str(e))
        sys.exit(3)

if __name__ == "__main__":
    main()
