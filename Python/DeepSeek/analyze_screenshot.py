# analyze_screenshot.py
# Analiza una imagen local con DeepSeek-VL (CPU) y devuelve JSON en consola.

import os, sys, json, re
from typing import Optional, Dict, Any, List

from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
from transformers import AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/deepseek-vl-1.3b-chat"

_proc = None
_tok = None
_model = None

def _ensure_model():
    global _proc, _tok, _model
    if _model is not None:
        return
    print(f"[*] Cargando modelo: {MODEL_NAME} (la primera vez puede tardar)...")
    _proc = VLChatProcessor.from_pretrained(MODEL_NAME)
    _tok = _proc.tokenizer
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    ).eval()
    print("[*] Modelo cargado.")

def _default_prompt() -> str:
    return (
        "<image_placeholder>"
        "Describe la captura. Necesito:"
        " 1) ¿Es juego, aplicación o escritorio inactivo?"
        " 2) Nombre del programa/juego si se reconoce."
        " 3) Si hay código, ¿en qué lenguaje?"
        " Responde primero con una breve descripción y luego, si puedes, un JSON con {actividad, app, lenguaje}."
    )

def _heuristic_parse(text: str) -> Dict[str, Optional[str]]:
    """
    Intenta extraer {actividad, app, lenguaje} del texto del modelo.
    1) Si encuentra un JSON válido, lo usa.
    2) Si no, usa heurísticas simples.
    """
    out = {"actividad": None, "app": None, "lenguaje": None}

    # 1) ¿Vino JSON?
    # Busca el primer bloque {...} y trata de parsearlo
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            j = json.loads(m.group(0))
            out["actividad"] = j.get("actividad") or j.get("activity") or j.get("estado")
            out["app"] = j.get("app") or j.get("programa") or j.get("software")
            out["lenguaje"] = j.get("lenguaje") or j.get("language")
            if any(v is not None for v in out.values()):
                return out
    except Exception:
        pass

    # 2) Heurísticas
    low = text.lower()

    # actividad
    if any(w in low for w in ["juego", "gaming", "gameplay"]):
        out["actividad"] = "juego"
    elif any(w in low for w in ["código", "programación", "programming", "source code"]):
        out["actividad"] = "programación"
    elif any(w in low for w in ["escritorio inactivo", "sin actividad", "idle", "escritorio", "desktop"]):
        out["actividad"] = out["actividad"] or "escritorio inactivo"
    else:
        if any(w in low for w in ["app", "aplicación", "window", "ventana", "ui", "interfaz"]):
            out["actividad"] = "aplicación"

    # app conocida (lista corta, añade lo que quieras)
    known_apps = [
        "Visual Studio Code", "VS Code", "Android Studio", "IntelliJ", "Eclipse",
        "PyCharm", "Notepad++", "Sublime", "Chrome", "Edge", "Firefox",
        "Word", "Excel", "PowerPoint", "Unity", "Unreal", "Blender"
    ]
    for name in known_apps:
        if name.lower() in low:
            out["app"] = name
            break

    # lenguaje
    lang_map = {
        "python": "Python", "java": "Java", "c#": "C#", "c++": "C++",
        "javascript": "JavaScript", "typescript": "TypeScript",
        "php": "PHP", "go": "Go", "rust": "Rust",
        "kotlin": "Kotlin", "swift": "Swift", "dart": "Dart"
    }
    for k, v in lang_map.items():
        if re.search(rf"\b{k}\b", low):
            out["lenguaje"] = v
            if out["actividad"] is None:
                out["actividad"] = "programación"
            break

    return out

def analyze(image_path: str, question: Optional[str] = None) -> Dict[str, Any]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No existe la imagen: {image_path}")
    _ensure_model()

    prompt = _default_prompt() if not question else f"<image_placeholder>{question.strip()}"
    convo = [
        {"role": "User", "content": prompt, "images": [image_path]},
        {"role": "Assistant", "content": ""}
    ]

    pil_images = load_pil_images(convo)
    inputs = _proc(conversations=convo, images=pil_images, force_batchify=True)

    out_tokens = _model.generate(
        **_model.prepare_inputs_for_generation(**inputs),
        max_new_tokens=256,
        do_sample=False
    )
    text = _tok.decode(out_tokens[0], skip_special_tokens=True).strip()

    parsed = _heuristic_parse(text)
    return {
        "actividad": parsed["actividad"],
        "app": parsed["app"],
        "lenguaje": parsed["lenguaje"],
        "raw_text": text
    }

def _find_local_image(candidates: List[str] = None) -> Optional[str]:
    if candidates is None:
        candidates = ["screenshot.png", "screenshot.jpg", "screenshot.jpeg",
                      "captura.png", "captura.jpg", "captura.jpeg"]
    # busca cualquiera .png/.jpg en la carpeta
    for name in candidates:
        if os.path.isfile(name):
            return name
    for fn in os.listdir("."):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            return fn
    return None

if __name__ == "__main__":
    # Usa la imagen en el mismo directorio.
    img = sys.argv[1] if len(sys.argv) > 1 else _find_local_image()
    if not img:
        print("No encontré una imagen en esta carpeta.\n"
              "Pon tu imagen (ej. screenshot.png) junto a este .py o pásala como argumento:\n"
              "  python analyze_screenshot.py mi_captura.png")
        sys.exit(1)

    try:
        result = analyze(img)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False, indent=2))
        sys.exit(1)
