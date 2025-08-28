# deepseek_local.py
# Analizador local de screenshots con DeepSeek-VL (CPU)
# Devuelve JSON {actividad, app, lenguaje, raw_text}

import os
import json
from typing import Optional, Dict, Any

from deepseek_vl.models import VLChatProcessor
from deepseek_vl.utils.io import load_pil_images
from transformers import AutoModelForCausalLM

_MODEL_NAME = "deepseek-ai/deepseek-vl-1.3b-chat"
_proc = None
_tok = None
_model = None

def _ensure_model():
    """Carga perezosa del modelo (una sola vez)."""
    global _proc, _tok, _model
    if _model is not None:
        return
    print(f"[*] Cargando modelo: {_MODEL_NAME} (la primera vez puede tardar)...")
    _proc = VLChatProcessor.from_pretrained(_MODEL_NAME)
    _tok = _proc.tokenizer
    # trust_remote_code=True porque la repo define helpers propios
    _model = AutoModelForCausalLM.from_pretrained(
        _MODEL_NAME, trust_remote_code=True
    ).eval()
    print("[*] Modelo cargado.")

def _build_prompt(custom_question: Optional[str]) -> str:
    if custom_question and custom_question.strip():
        return f"<image_placeholder>{custom_question.strip()}"
    # Pregunta fija por defecto (tu caso de uso)
    return (
        "<image_placeholder>"
        "Describe la captura. Necesito: "
        "1) ¿Es juego, app o escritorio inactivo? "
        "2) Nombre del programa/juego si se reconoce. "
        "3) Si hay código, ¿en qué lenguaje?"
    )

def _postprocess_to_json(text: str) -> Dict[str, Any]:
    """
    Intenta extraer estructura. Si el modelo no entrega JSON, mapea heurísticamente.
    """
    out = {
        "actividad": None,  # "juego" | "aplicación" | "escritorio inactivo" | "programación" | etc.
        "app": None,
        "lenguaje": None,
        "raw_text": text.strip()
    }
    # Intento 1: ya vino JSON
    try:
        j = json.loads(text)
        # normaliza claves más comunes
        out["actividad"] = j.get("actividad") or j.get("activity") or j.get("estado")
        out["app"] = j.get("app") or j.get("programa") or j.get("software")
        out["lenguaje"] = j.get("lenguaje") or j.get("language") or j.get("idioma_codigo")
        # rellena raw_text con el JSON bonito
        out["raw_text"] = json.dumps(j, ensure_ascii=False)
        return out
    except Exception:
        pass

    # Intento 2: extraer por texto libre (muy simple, ajusta si quieres)
    low = text.lower()
    if any(w in low for w in ["juego", "game", "gaming"]):
        out["actividad"] = out["actividad"] or "juego"
    if any(w in low for w in ["visual studio code", "vscode", "vs code", "android studio", "intellij",
                              "eclipse", "pycharm", "sublime", "notepad++", "chrome", "edge", "word", "excel"]):
        # captura un nombre típico de app si aparece
        for name in ["Visual Studio Code", "VS Code", "Android Studio", "IntelliJ", "Eclipse",
                     "PyCharm", "Sublime", "Notepad++", "Chrome", "Edge", "Word", "Excel"]:
            if name.lower() in low:
                out["app"] = name
                break
        out["actividad"] = out["actividad"] or "aplicación"
    if any(w in low for w in ["escritorio", "desktop", "idle", "inactivo", "sin actividad"]):
        out["actividad"] = out["actividad"] or "escritorio inactivo"
    # lenguajes comunes
    langs = ["python", "java", "c#", "c++", "javascript", "typescript", "php", "go", "rust", "kotlin", "swift"]
    for lg in langs:
        if lg in low:
            out["lenguaje"] = lg.capitalize() if lg != "c#" else "C#"
            # si detectamos lenguaje, es razonable asumir "programación"
            out["actividad"] = out["actividad"] or "programación"
            break
    return out

def analyze(image_path: str, question: Optional[str] = None) -> Dict[str, Any]:
    """
    Analiza una imagen/screenshot y devuelve:
      {
        "actividad": "...",
        "app": "...",
        "lenguaje": "...",
        "raw_text": "respuesta completa del modelo"
      }
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No existe la imagen: {image_path}")
    _ensure_model()

    convo = [{
        "role": "User",
        "content": _build_prompt(question),
        "images": [image_path]
    }, {"role": "Assistant", "content": ""}]

    pil_images = load_pil_images(convo)
    inputs = _proc(conversations=convo, images=pil_images, force_batchify=True)

    # La implementación del modelo expone prepare_inputs_for_generation via trust_remote_code
    out_tokens = _model.generate(
        **_model.prepare_inputs_for_generation(**inputs),
        max_new_tokens=256,
        do_sample=False
    )
    text = _tok.decode(out_tokens[0], skip_special_tokens=True)
    return _postprocess_to_json(text)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyzer local con DeepSeek-VL (CPU).")
    parser.add_argument("image", help="Ruta a PNG/JPG con el screenshot")
    parser.add_argument("-q", "--question", help="Pregunta personalizada (opcional)", default=None)
    args = parser.parse_args()

    result = analyze(args.image, args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))
