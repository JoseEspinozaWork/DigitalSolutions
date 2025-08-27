import os, sys, json, base64, requests

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
IMG_PATH_ARG = sys.argv[1] if len(sys.argv) > 1 else None

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

def call_generate(img_b64: str):
    url = f"{OLLAMA}/api/generate"
    payload = {
        "model": "llava:7b",
        "prompt": """Analiza la captura de pantalla y descríbela en un párrafo.
Indica claramente:
- si es código, un juego, una aplicación de oficina o solo escritorio inactivo,
- si es código, en qué lenguaje parece estar escrito,
- si puedes, menciona el entorno (Visual Studio Code, IntelliJ, Word, Excel, navegador, etc.),
- describe lo que parece estar haciendo el usuario (editando, depurando, jugando, navegando, etc.).
Responde en español, en un solo párrafo, sin listas ni JSON."""
,
        "images": [img_b64],
        "stream": False
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()

def call_chat(img_b64: str):
    url = f"{OLLAMA}/api/chat"
    # Algunas versiones requieren images como objetos { data: ... }
    payload = {
        "model": "llava:7b",
        "messages": [
            {"role": "system", "content": "Eres un asistente que describe imágenes."},
            {"role": "user", "content": "Describe esta imagen en detalle.", "images": [ { "data": img_b64 } ]}
        ],
        "stream": False
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
        print("Asegúrate de tener Ollama abierto y de ver modelos en /api/tags.")
        sys.exit(1)
    else:
        print("[OK] Servidor activo.")

    try:
        img_path = pick_image_path()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(2)

    print(f"[INFO] Leyendo imagen: {img_path}")
    img_b64 = read_b64(img_path)

    # 1) Intento con /api/generate
    try:
        print("[INFO] Llamando /api/generate ...")
        data = call_generate(img_b64)
        print("\n=== JSON COMPLETO (/api/generate) ===")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:4000])
        print("\n--- SOLO TEXTO ---\n")
        print(data.get("response", ""))
        return
    except requests.HTTPError as he:
        txt = he.response.text if he.response is not None else str(he)
        print("[WARN] /api/generate falló. Intentaré /api/chat.")
        print("Detalles:", txt[:500])
    except Exception as e:
        print("[WARN] /api/generate falló (excepción). Intentaré /api/chat.")
        print("Detalles:", str(e))

    # 2) Fallback con /api/chat
    try:
        print("[INFO] Llamando /api/chat ...")
        data = call_chat(img_b64)
        print("\n=== JSON COMPLETO (/api/chat) ===")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:4000])
        print("\n--- SOLO TEXTO ---\n")
        msg = (data.get("message") or {}).get("content", "")
        print(msg)
    except Exception as e:
        print("[ERROR] También falló /api/chat.")
        print("Detalles:", str(e))
        sys.exit(3)

if __name__ == "__main__":
    main()
