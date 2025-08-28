import os, sys, time, json, base64, queue, threading, requests, re
from io import BytesIO
from PIL import Image

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL  = os.environ.get("VISION_MODEL", "llava:7b")  # cambia a llava:13b si quieres
MAX_WIDTH = 1024            # redimensionar para velocidad
SIM_THRESH = 3.0            # umbral de diferencia (%) para considerar "igual"
POLL_SECONDS = 60           # cada cuánto entra una imagen (tu caso)
TIMEOUT_SEC = 300

def enforce_brief(text: str, max_sentences: int = 2) -> str:
    if not text: return ""
    t = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    brief = " ".join(parts[:max_sentences])
    if brief and brief[-1] not in ".!?":
        brief += "."
    return brief

def to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def load_and_downscale(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if im.width > MAX_WIDTH:
        h = int(im.height * (MAX_WIDTH / im.width))
        im = im.resize((MAX_WIDTH, h), Image.LANCZOS)
    return im

def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    # compara versiones pequeñas en escala de grises
    s = (160, int(160 * a.height / a.width)) if a.width else (160, 90)
    a2 = a.resize(s, Image.BILINEAR).convert("L")
    b2 = b.resize(s, Image.BILINEAR).convert("L")
    pa = a2.tobytes()
    pb = b2.tobytes()
    total = sum(abs(pa[i] - pb[i]) for i in range(len(pa)))
    return 100.0 * total / (255.0 * len(pa))

def describe_image_b64(img_b64: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Responde en español, conciso (1–2 frases), sin listas ni saltos de línea."},
            {"role": "user",
             "content": ("En 1–2 frases: di si la captura muestra código (lenguaje y entorno si lo reconoces), "
                         "un juego, app de oficina, navegador u otra app; y qué está haciendo el usuario."),
             "images": [img_b64]}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 128,     # más corto = más rápido
            "num_ctx": 1024         # contexto reducido
        }
    }
    r = requests.post(f"{OLLAMA}/api/chat", json=payload, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()
    text = (data.get("message") or {}).get("content", "")
    return enforce_brief(text)

# ===== Cola con "coalescing": solo 1 elemento =====
task_q = queue.Queue(maxsize=1)
last_image = None
last_result = None

def worker():
    global last_image, last_result
    while True:
        path = task_q.get()  # bloquea hasta que haya trabajo
        try:
            im = load_and_downscale(path)
            # si hay imagen previa, y son casi iguales, reutiliza resultado
            if last_image is not None:
                diff = mean_abs_diff(im, last_image)
                if diff < SIM_THRESH and last_result:
                    print(f"[SKIP] Cambios mínimos ({diff:.2f}%), reutilizo resultado:\n{last_result}\n")
                    continue
            b64 = to_b64(im)
            t0 = time.time()
            result = describe_image_b64(b64)
            dt = time.time() - t0
            print(f"[OK] {os.path.basename(path)} ({dt:.1f}s)\n{result}\n")
            last_image = im
            last_result = result
        except Exception as e:
            print("[ERROR]", e)
        finally:
            task_q.task_done()

def enqueue_latest(path: str):
    # Descarta lo que hubiera y mete la última imagen
    try:
        while True:
            task_q.get_nowait()
            task_q.task_done()
        # vaciamos todo lo pendiente
    except queue.Empty:
        pass
    try:
        task_q.put_nowait(path)
    except queue.Full:
        pass

def main_loop(paths_iterable):
    # Lanza el worker
    t = threading.Thread(target=worker, daemon=True)
    t.start()

    for path in paths_iterable:
        if not os.path.isfile(path):
            print(f"[WARN] No existe: {path}")
        else:
            enqueue_latest(path)
        time.sleep(POLL_SECONDS)  # simula tu llegada cada minuto

if __name__ == "__main__":
    # USO 1: pasar rutas por argumentos y que el bucle las "simule" por minuto
    #   python fast_loop.py img1.png img2.png ...
    # USO 2: integrar esto con tu capturador que va generando 'ultima.png' cada minuto.
    if len(sys.argv) > 1:
        main_loop(sys.argv[1:])
    else:
        # ejemplo mínimo: usa siempre la misma ruta (cámbiala por tu capturador)
        demo_path = input("Ruta de la imagen a monitorear cada minuto: ").strip('"').strip()
        def gen():
            while True:
                yield demo_path
        main_loop(gen())
