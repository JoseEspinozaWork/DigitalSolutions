#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DescriptorImagenes – CLIP + Ensamble (potente y preciso, 100% local)
--------------------------------------------------------------------
Objetivo: Dada una captura, decir si hay código, en qué lenguaje, y qué app/juego se ve.
Este pipeline prioriza **precisión** sobre velocidad. Si algún modelo falta, 
se degrada de forma elegante sin romperse.

Componentes (todos open-source):
  • OCR: EasyOCR (preferido) → Tesseract (fallback)
  • Lenguaje: Ensamble = Tree‑sitter (si está) + Heurística/Pygments + (opcional) CodeBERT/Zero‑Shot
  • App/Juego: CLIP (si está) + heurística por texto (fallback)

Instalación sugerida (activa sólo lo que uses):
  pip install opencv-python pillow pygments pytesseract
  pip install easyocr
  pip install transformers torch --upgrade
  pip install tree_sitter_languages
  # (CLIP/BLIP vienen con transformers; los pesos se descargan en el primer uso)

Uso:
  python descriptor_clip_plus.py --image ruta/a/captura.png \
      --enable-clip --enable-treesitter --prefer-easyocr

Pruebas (si creaste las imágenes de prueba):
  python descriptor_clip_plus.py --folder /mnt/data/descriptor_tests --enable-clip --enable-treesitter --prefer-easyocr --save-json /mnt/data/out.json

"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# ---------------------------
# Cargas opcionales con degradación elegante
# ---------------------------

def try_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None

pytesseract = try_import('pytesseract')
easyocr = try_import('easyocr')
pygments = try_import('pygments')
if pygments:
    from pygments.lexers import guess_lexer
    from pygments.util import ClassNotFound
transformers = try_import('transformers')
torch = try_import('torch')

# Tree‑sitter (paquete tree_sitter_languages)
_ts_langs = try_import('tree_sitter_languages')
if _ts_langs:
    from tree_sitter_languages import get_language, get_parser

# ---------------------------
# Configuración
# ---------------------------

LANGS = [
    "csharp","vbnet","java","python","javascript","typescript",
    "cpp","go","php","ruby","sql","shell"
]

# Gramáticas disponibles en tree_sitter_languages (pueden variar por entorno)
TS_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "cpp": "cpp",
    "go": "go",
    "php": "php",
    "ruby": "ruby",
    "csharp": "c_sharp",   # puede faltar en algunos builds
    # sql/shell/vbnet típicamente no disponibles
}

# Lista de apps/juegos candidatas para CLIP y heurística por texto
KNOWN_APPS = [
    "Visual Studio", "Visual Studio Code", "MySQL Workbench", "SSMS", "SQL Server Management Studio",
    "IntelliJ IDEA", "PyCharm", "Android Studio", "Sublime Text", "Rider", "Xcode",
    "Google Chrome", "Microsoft Edge", "Mozilla Firefox", "Brave", "Opera",
    "Postman", "Insomnia", "GitHub Desktop", "Docker Desktop", "PowerShell", "Terminal",
    "Photoshop", "Illustrator", "GIMP", "Inkscape",
    "Unity", "Unreal Engine", "Minecraft", "Fortnite", "League of Legends", "Steam"
]

# Palabras clave por lenguaje (reforzadas para C#/VB.NET)
LANG_KEYWORDS = {
    "csharp": {
        "using ", "namespace ", "public ", "class ", "string ", "async ", "await ",
        "IEnumerable<", "Console.WriteLine", "var ", "MySqlConnection", "DbContext", "Task<", "List<"
    },
    "vbnet": {"Public ", "Sub ", "Function ", "End Sub", "End Function", "Dim ", "As ", "Imports "},
    "java": {"public ", "class ", "static ", "void ", "new ", "extends ", "implements "},
    "python": {"def ", "import ", "self", "None", "True", "False", "lambda ", "async ", "await "},
    "javascript": {"function ", "const ", "let ", "=>", "document", "console", "import ", "export "},
    "typescript": {"interface ", "implements ", "enum ", "type ", "readonly ", "as ", "import "},
    "cpp": {"#include", "std::", "template ", "cout", "cin", "::"},
    "go": {"package ", "func ", "fmt.", "defer ", "go ", "chan ", "struct "},
    "php": {"<?php", " echo ", "$this", "use ", "namespace ", "->"},
    "ruby": {"def ", " end", "class ", ":", "module ", "puts"},
    "sql": {"SELECT ", "FROM ", "WHERE ", "JOIN ", "INSERT ", "UPDATE ", "DELETE ", "INNER JOIN", "LEFT JOIN"},
    "shell": {"#!/bin/bash", "#!/usr/bin/env", " sudo ", " grep ", " awk ", " sed "},
}

CODE_SIGNALS = ["{", "}", ";", "==", "!=", "<=", ">=", "->", "::", "#include", "#!/", "def "]

# Priors por aplicación detectada (sube pesos de lenguajes más probables)
APP_LANGUAGE_PRIORS = {
    "Visual Studio": {"csharp": 2.5, "vbnet": 1.5, "cpp": 1.0},
    "Visual Studio Code": {"javascript": 1.8, "typescript": 1.8, "python": 1.5, "csharp": 1.2, "cpp": 1.0},
    "MySQL Workbench": {"sql": 3.0},
    "SSMS": {"sql": 3.0},
    "SQL Server Management Studio": {"sql": 3.0},
    "PyCharm": {"python": 3.0},
    "IntelliJ IDEA": {"java": 2.0, "kotlin": 1.5},
    "Android Studio": {"java": 1.8, "kotlin": 1.8},
    "Rider": {"csharp": 2.0},
    "Xcode": {"cpp": 1.5},
    "Postman": {"javascript": 1.2},
    "PowerShell": {"shell": 2.5},
}

# ---------------------------
# Utilidades
# ---------------------------

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    return img

def resize_max(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ---------------------------
# OCR (EasyOCR preferido, Tesseract fallback)
# ---------------------------

@dataclass
class OCRResult:
    text: str
    words: List[str]
    density: float  # palabras por megapixel

def run_ocr(img: np.ndarray, prefer_easyocr: bool = True) -> OCRResult:
    h, w = img.shape[:2]
    mpix = max(1e-6, (h * w) / 1_000_000.0)

    if prefer_easyocr and easyocr is not None:
        try:
            reader = easyocr.Reader(['en'], gpu=(torch is not None and torch.cuda.is_available()))
            result = reader.readtext(img)
            words = []
            for _, text, conf in result:
                if text and conf >= 0.4:
                    words.extend(text.strip().split())
            return OCRResult(text=' '.join(words), words=words, density=len(words)/mpix)
        except Exception:
            pass

    if pytesseract is None:
        return OCRResult(text='', words=[], density=0.0)
    try:
        data = pytesseract.image_to_data(to_pil(img), output_type=pytesseract.Output.DICT)
        words = []
        for wtxt, c in zip(data.get('text', []), data.get('conf', [])):
            try:
                c = float(c)
            except Exception:
                c = -1
            if wtxt and wtxt.strip() and c > 40:
                words.append(wtxt.strip())
        return OCRResult(text=' '.join(words), words=words, density=len(words)/mpix)
    except Exception:
        return OCRResult(text='', words=[], density=0.0)

# ---------------------------
# Lenguaje: Heurística + Pygments
# ---------------------------

def heuristic_language(text: str) -> Tuple[Optional[str], Dict[str, int], Dict[str, Any]]:
    if not text or len(text) < 10:
        return None, {}, {"has_code": False}

    signal_hits = sum(1 for s in CODE_SIGNALS if s in text)
    tokens = set(t for t in text.replace('\n', ' ').split(' ') if t)

    scores: Dict[str, int] = {}
    for lang, keys in LANG_KEYWORDS.items():
        hits = sum(1 for k in keys if any(k in t for t in tokens))
        scores[lang] = hits

    best = max(scores, key=scores.get) if scores else None

    pyg_guess = None
    if pygments:
        try:
            lexer = guess_lexer(text)
            pyg_guess = lexer.name.lower()
        except Exception:
            pyg_guess = None

    # Normaliza Pygments a nuestras etiquetas
    map_pyg = {
        "python":"python","javascript":"javascript","typescript":"typescript","c#":"csharp","c++":"cpp","java":"java",
        "kotlin":"kotlin","php":"php","ruby":"ruby","sql":"sql","bash":"shell","powershell":"shell","vb":"vbnet"
    }
    pyg_lang = None
    if pyg_guess:
        for k,v in map_pyg.items():
            if k in pyg_guess:
                pyg_lang = v
                break

    has_code = (signal_hits >= 2) or (scores.get(best,0) >= 3) or (pyg_lang is not None)
    lang = best or pyg_lang
    return lang, scores, {"has_code": has_code, "signal_hits": signal_hits, "pygments": pyg_guess}

# ---------------------------
# Lenguaje: Tree‑sitter (score parseable)
# ---------------------------

def ts_score(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not _ts_langs or not text or len(text) < 10:
        return out
    for lang, tsname in TS_MAP.items():
        try:
            parser = get_parser(tsname)  # ya viene configurado
            tree = parser.parse(text.encode('utf8'))
            root = tree.root_node
            # Si el árbol no tiene errores → score alto, si tiene → bajo
            ok = False
            try:
                ok = not root.has_error
            except Exception:
                try:
                    ok = not root.has_error()
                except Exception:
                    ok = True
            out[lang] = 0.95 if ok else 0.35
        except Exception:
            continue
    return out

# ---------------------------
# Lenguaje: CodeBERT (opcional, señal débil)
# ---------------------------

def codebert_scores(text: str) -> Dict[str, float]:
    if transformers is None or not text or len(text) < 10:
        return {}
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        model_name = os.environ.get('CODE_MODEL', 'microsoft/codebert-base')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(LANGS))
        device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
        model.to(device)
        inputs = tokenizer(text[:1024], return_tensors='pt', truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = logits.softmax(dim=-1).detach().cpu().numpy()[0]
        return {lang: float(probs[i]) for i,lang in enumerate(LANGS)}
    except Exception:
        return {}

# ---------------------------
# App/Juego: CLIP (opcional) y heurística por texto
# ---------------------------

def simple_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    matches = sum(1 for ch in a if ch in b)
    return 2.0 * matches / (len(a)+len(b)+1e-6)


def app_from_text(words: List[str]) -> Optional[Dict[str, Any]]:
    if not words:
        return None
    hay = " ".join(words)
    best, best_s = None, 0.0
    for app in KNOWN_APPS:
        s = simple_similarity(hay, app)
        if s > best_s:
            best, best_s = app, s
    if best and best_s >= 0.3:
        return {"name": best, "score": round(best_s, 2)}
    return None


def app_from_clip(img: np.ndarray) -> Optional[Dict[str, Any]]:
    if transformers is None:
        return None
    try:
        from transformers import CLIPProcessor, CLIPModel
        device = 'cuda' if (torch is not None and torch.cuda.is_available()) else 'cpu'
        model_name = os.environ.get('CLIP_MODEL', 'openai/clip-vit-base-patch32')
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device)
        inputs = processor(text=KNOWN_APPS, images=to_pil(img), return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
            probs = out.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        return {"name": KNOWN_APPS[idx], "score": float(probs[idx])}
    except Exception:
        return None

# ---------------------------
# Ensamble de lenguaje con priors por aplicación
# ---------------------------

def pick_language(text: str, app_name: Optional[str]) -> Dict[str, Any]:
    result = {"final": None, "scores": {}, "details": {}}
    if not text or len(text) < 10:
        return result

    # 1) Heurística
    h_lang, h_scores, h_meta = heuristic_language(text)
    base = {k: float(v) for k,v in h_scores.items()}
    result["details"]["heuristic_meta"] = h_meta

    # 2) Tree‑sitter
    ts = ts_score(text)
    for k,v in ts.items():
        base[k] = base.get(k, 0.0) + 5.0 * v  # peso fuerte a parseabilidad
    result["details"]["tree_sitter"] = ts

    # 3) CodeBERT (débil)
    cb = codebert_scores(text)
    for k,v in cb.items():
        base[k] = base.get(k, 0.0) + 0.8 * v
    if cb:
        result["details"]["codebert"] = cb

    # 4) Priors por aplicación (si la hay)
    if app_name and app_name in APP_LANGUAGE_PRIORS:
        pri = APP_LANGUAGE_PRIORS[app_name]
        for k, mult in pri.items():
            base[k] = base.get(k, 0.0) * mult
        result["details"]["app_priors"] = {app_name: pri}

    # Normaliza y elige
    if not base:
        return result
    # pequeña regularización
    for k in LANGS:
        base[k] = base.get(k, 0.0)
    # decisión final
    final = max(base, key=base.get)
    result["final"] = final
    result["scores"] = base
    result["details"]["heuristic_best"] = h_lang
    return result

# ---------------------------
# Descripción principal
# ---------------------------

@dataclass
class Description:
    type: str
    summary: str
    details: Dict[str, Any]


def describe_image(path: str, enable_clip: bool = True, enable_treesitter: bool = True, prefer_easyocr: bool = True) -> Description:
    img = resize_max(load_image(path))

    # 1) OCR
    ocr = run_ocr(img, prefer_easyocr=prefer_easyocr)

    # 2) App
    app_info = app_from_clip(img) if enable_clip else None
    if not app_info:
        app_info = app_from_text(ocr.words)
    app_name = app_info["name"] if app_info else None

    # 3) Lenguaje (con priors de app)
    lang_vote = pick_language(ocr.text, app_name)
    language = lang_vote.get('final')

    # 4) Tipo final
    if language:
        summary = f"Se detecta código ({language})."
        if app_name:
            summary += f" Aplicación probable: {app_name}."
        return Description(
            type="code",
            summary=summary,
            details={
                "language": language,
                "app": app_info,
                "ocr_text_density": ocr.density,
                "votes": lang_vote,
            }
        )

    if app_info:
        return Description(
            type="application",
            summary=f"Se detecta una aplicación: {app_info['name']} (score {round(app_info['score'],2)}).",
            details={
                "app": app_info,
                "ocr_text_density": ocr.density,
            }
        )

    # Heurística de inactividad
    return Description(
        type="unknown",
        summary="No se pudo clasificar con alta confianza.",
        details={
            "ocr_text_density": ocr.density,
        }
    )

# ---------------------------
# Salidas human‑friendly (pretty / markdown)
# ---------------------------

def _top_keywords(scores: Dict[str, float], n: int = 3) -> List[Tuple[str,float]]:
    if not scores:
        return []
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]


def render_pretty(result: Dict[str, Any]) -> str:
    # result es un dict convertido desde Description (asdict)
    t = result.get('type')
    details = result.get('details', {})
    lines = []
    if t == 'code':
        lang = details.get('language') or 'desconocido'
        app = (details.get('app') or {}).get('name') if details.get('app') else None
        lines.append(f"➡️  Se detecta **código** en **{lang}**.")
        if app:
            lines.append(f"➡️  Aplicación probable: **{app}**.")
        # Evidencias
        votes = details.get('votes', {})
        if isinstance(votes, dict):
            scores = votes.get('scores', {})
            if scores:
                top3 = _top_keywords(scores, 3)
                pretty_scores = ", ".join([f"{k}: {round(v,2)}" for k,v in top3])
                lines.append(f"🔍 Evidencia (top): {pretty_scores}")
            ts = votes.get('details', {}).get('tree_sitter') if isinstance(votes.get('details'), dict) else None
            if ts:
                top_ts = _top_keywords(ts, 3)
                if top_ts:
                    pretty_ts = ", ".join([f"{k}: {round(v,2)}" for k,v in top_ts])
                    lines.append(f"🌲 Tree‑sitter: {pretty_ts}")
        dens = details.get('ocr_text_density')
        if dens is not None:
            lines.append(f"🅾️  Densidad OCR: {dens:.2f} palabras/MPix")
        # Comentarios y recomendaciones
        lines.append("💬 Comentario: Si el lenguaje no coincide con lo esperado, activa --enable-treesitter y --enable-clip para mejorar la precisión, y asegúrate de usar EasyOCR con --prefer-easyocr.")
    elif t == 'application':
        app = (details.get('app') or {}).get('name') if details.get('app') else 'desconocida'
        score = (details.get('app') or {}).get('score') if details.get('app') else None
        lines.append(f"➡️  Se detecta **aplicación**: **{app}**" + (f" (score {round(score,2)})" if score is not None else "") + ".")
        dens = details.get('ocr_text_density')
        if dens is not None:
            lines.append(f"🅾️  Densidad OCR: {dens:.2f} palabras/MPix")
        lines.append("💬 Comentario: Para identificar el lenguaje de código, asegúrate de que haya texto legible en la captura (OCR) y habilita el ensamble de lenguaje.")
    else:
        lines.append("➡️  No se pudo clasificar con alta confianza.")
        dens = details.get('ocr_text_density')
        if dens is not None:
            lines.append(f"🅾️  Densidad OCR: {dens:.2f} palabras/MPix")
        lines.append("💬 Comentario: Aumenta el tamaño de fuente de la captura, evita fondos borrosos y activa --prefer-easyocr para mejorar el OCR.")
    return "".join(lines)


def render_markdown(result: Dict[str, Any]) -> str:
    # Parecido a pretty, pero con encabezados
    t = result.get('type')
    details = result.get('details', {})
    out = ["# Resultado de DescriptorImagenes"]
    if t == 'code':
        lang = details.get('language') or 'desconocido'
        app = (details.get('app') or {}).get('name') if details.get('app') else None
        out.append(f"- **Tipo**: Código")
        out.append(f"- **Lenguaje**: **{lang}**")
        if app:
            out.append(f"- **Aplicación**: **{app}**")
        votes = details.get('votes', {})
        if isinstance(votes, dict):
            scores = votes.get('scores', {})
            if scores:
                top3 = _top_keywords(scores, 3)
                out.append("- **Evidencia (top)**: " + ", ".join([f"{k}: {round(v,2)}" for k,v in top3]))
            ts = votes.get('details', {}).get('tree_sitter') if isinstance(votes.get('details'), dict) else None
            if ts:
                top_ts = _top_keywords(ts, 3)
                if top_ts:
                    out.append("- **Tree‑sitter**: " + ", ".join([f"{k}: {round(v,2)}" for k,v in top_ts]))
        dens = details.get('ocr_text_density')
        if dens is not None:
            out.append(f"- **Densidad OCR**: {dens:.2f} palabras/MPix")
        out.append("**Comentario**: Si el lenguaje no coincide con lo esperado, activa `--enable-treesitter` y `--enable-clip`, y usa `--prefer-easyocr`.")
    elif t == 'application':
        app = (details.get('app') or {}).get('name') if details.get('app') else 'desconocida'
        score = (details.get('app') or {}).get('score') if details.get('app') else None
        out.append(f"- **Tipo**: Aplicación")
        out.append(f"- **Aplicación**: **{app}**" + (f" (score {round(score,2)})" if score is not None else ""))
        dens = details.get('ocr_text_density')
        if dens is not None:
            out.append(f"- **Densidad OCR**: {dens:.2f} palabras/MPix")
        out.append("**Comentario**: Para detectar el lenguaje, habilita OCR y el ensamble de lenguaje.")
    else:
        out.append(f"- **Tipo**: Indefinido")
        dens = details.get('ocr_text_density')
        if dens is not None:
            out.append(f"- **Densidad OCR**: {dens:.2f} palabras/MPix")
        out.append("**Comentario**: Sube una captura con más texto y activa `--prefer-easyocr`.")
    return "".join(out)

# ---------------------------
# CLI
# ---------------------------

def run_on_path(path: str, args) -> Dict[str, Any]:
    desc = describe_image(path, enable_clip=args.enable_clip, enable_treesitter=args.enable_treesitter, prefer_easyocr=args.prefer_easyocr)
    out = asdict(desc)
    out["image"] = path
    return out


def main():
    p = argparse.ArgumentParser(description="DescriptorImagenes – CLIP + Ensamble")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--image', type=str, help='Ruta a una imagen')
    g.add_argument('--folder', type=str, help='Ruta a una carpeta con imágenes')
    p.add_argument('--enable-clip', action='store_true', help='Usar CLIP para detectar aplicación')
    p.add_argument('--enable-treesitter', action='store_true', help='Usar Tree‑sitter para puntuar lenguajes')
    p.add_argument('--prefer-easyocr', action='store_true', help='Usar EasyOCR antes que Tesseract')
    p.add_argument('--format', choices=['json','pretty','markdown'], default='json', help='Formato de salida legible')
    p.add_argument('--save-json', type=str, help='Guardar resultados crudos en JSON')
    args = p.parse_args()

    results = []
    if args.image:
        results.append(run_on_path(args.image, args))
    else:
        for fname in sorted(os.listdir(args.folder)):
            if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp','.webp')):
                results.append(run_on_path(os.path.join(args.folder, fname), args))

    # Persistencia cruda si se solicita
    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Resultados guardados en: {args.save_json}")

    # Render humano
    if args.format == 'json' or len(results) > 1:
        # Si hay varias imágenes, siempre imprimimos JSON para evitar confusiones,
        # pero mostramos un resumen legible por cada entrada
        if args.format in ('pretty','markdown'):
            for r in results:
                txt = render_pretty(r) if args.format=='pretty' else render_markdown(r)
                print("" + ("-"*60) + "" + txt)
        print(json.dumps(results if len(results)>1 else results[0], ensure_ascii=False, indent=2))
    else:
        r = results[0]
        txt = render_pretty(r) if args.format=='pretty' else render_markdown(r)
        print(txt)

if __name__ == '__main__':
    main()
