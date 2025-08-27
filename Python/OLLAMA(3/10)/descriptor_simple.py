#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototipo mínimo y funcional para DescriptorImagenes (versión corregida)
-----------------------------------------------------------------------
Objetivo: Dada una captura de pantalla (imagen), producir una descripción simple:
 - Si hay código: detectarlo e intentar identificar el lenguaje.
 - Si hay una app/juego: intentar reconocer el software por texto visible (OCR) con coincidencia difusa.
 - Si no hay actividad relevante: reportarlo.

Dependencias (Python):
  pip install opencv-python pillow pytesseract pygments

Dependencia del sistema:
  - Tesseract OCR instalado y disponible en PATH.
    * Windows (choco):   choco install tesseract
    * Ubuntu/Debian:     sudo apt-get install tesseract-ocr
    * macOS (brew):      brew install tesseract

Uso:
  python descriptor_simple.py ruta/a/captura.png

Notas:
 - Esta versión elimina `rapidfuzz` para evitar errores de entorno.
 - Se usa una coincidencia difusa simple implementada manualmente (ratio de similitud).
 - El diseño sigue siendo modular y escalable.
"""

import sys
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
import pytesseract

# Pygments para heurística de lenguaje de programación
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

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


# ---------------------------
# OCR
# ---------------------------

@dataclass
class OCRResult:
    text: str
    words: List[str]
    conf: List[float]
    density: float  # palabras por megapixel


def run_ocr(img: np.ndarray) -> OCRResult:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
    words = []
    conf = []
    for w, c in zip(data.get('text', []), data.get('conf', [])):
        try:
            c = float(c)
        except Exception:
            c = -1
        if w and w.strip() and c > 40:  # filtrar ruido de bajo confidence
            words.append(w.strip())
            conf.append(c)
    text = " ".join(words)
    h, w = img.shape[:2]
    mpix = max(1e-6, (h * w) / 1_000_000.0)
    density = len(words) / mpix
    return OCRResult(text=text, words=words, conf=conf, density=density)


# ---------------------------
# Heurísticas de detección de ventanas / actividad
# ---------------------------

def estimate_window_rects(img: np.ndarray) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_like = 0
    h, w = img.shape[:2]
    area_img = h * w
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            area = cv2.contourArea(c)
            if area > 0.01 * area_img:
                rect_like += 1
    return rect_like


# ---------------------------
# Detección de código y lenguaje
# ---------------------------

LANG_KEYWORDS = {
    "python": {"def", "import", "self", "None", "True", "False", "lambda", "async", "await"},
    "javascript": {"function", "const", "let", "=>", "document", "console", "import", "export"},
    "typescript": {"interface", "implements", "enum", "type", "readonly", "as", "import"},
    "java": {"public", "class", "static", "void", "new", "extends", "implements"},
    "csharp": {"using", "namespace", "public", "class", "string", "async", "await"},
    "cpp": {"#include", "std::", "template", "cout", "cin", "int", "::"},
    "go": {"package", "func", "fmt.", "defer", "go ", "chan", "struct"},
    "php": {"<?php", "echo", "->$", "$this", "use ", "namespace"},
    "ruby": {"def ", "end", "class ", ":symbol", "module ", "puts"},
    "kotlin": {"fun ", "val ", "var ", "data class", ": String"},
    "vbnet": {"Public", "Sub", "Function", "End Sub", "End Function", "Dim ", "As ", "Imports"},
    "sql": {"SELECT", "FROM", "WHERE", "JOIN", "INSERT", "UPDATE", "DELETE"},
    "shell": {"#!/bin/bash", "#!/usr/bin/env", "sudo", "grep", "awk"},
}


def detect_code_and_language(ocr: OCRResult) -> Dict[str, Any]:
    text = ocr.text
    if not text or len(text) < 10:
        return {"has_code": False}

    code_signals = ["{", "}", ";", "==", "!=", "<=", ">=", "->", "::", "#include", "#!/", "def "]
    signal_hits = sum(1 for s in code_signals if s in text)

    guessed = None
    try:
        lexer = guess_lexer(text)
        guessed = lexer.name.lower()
    except ClassNotFound:
        guessed = None

    scores = {}
    tokens = set([t for t in text.replace("\n", " ").split(" ") if t])
    for lang, keys in LANG_KEYWORDS.items():
        hits = sum(1 for k in keys if any(k in t for t in tokens))
        scores[lang] = hits

    best_lang = max(scores, key=scores.get) if scores else None
    best_score = scores.get(best_lang, 0) if best_lang else 0

    has_code = (signal_hits >= 2) or (best_score >= 3) or (guessed is not None and any(x in guessed for x in ["python", "javascript", "java", "c#", "c++", "go", "php", "ruby", "kotlin", "sql", "bash", "powershell", "vb"]))

    language = None
    if has_code:
        if best_score >= 3:
            language = best_lang
        elif guessed:
            g = guessed
            if "python" in g: language = "python"
            elif "javascript" in g: language = "javascript"
            elif "typescript" in g: language = "typescript"
            elif "c#" in g: language = "csharp"
            elif "c++" in g: language = "cpp"
            elif "java" in g: language = "java"
            elif "kotlin" in g: language = "kotlin"
            elif "php" in g: language = "php"
            elif "ruby" in g: language = "ruby"
            elif "sql" in g: language = "sql"
            elif "bash" in g or "shell" in g: language = "shell"
            elif "vb" in g: language = "vbnet"

    return {
        "has_code": has_code,
        "language": language,
        "evidence": {
            "signal_hits": signal_hits,
            "keyword_score": best_score,
            "pygments": guessed,
        }
    }


# ---------------------------
# Detección de aplicaciones / juegos por texto
# ---------------------------

KNOWN_APPS = [
    "Visual Studio Code", "Visual Studio", "IntelliJ IDEA", "PyCharm", "Android Studio", "Sublime Text",
    "Notepad++", "Vim", "Emacs", "Rider", "Xcode",
    "Microsoft Word", "Microsoft Excel", "PowerPoint", "Outlook", "OneNote", "LibreOffice Writer", "LibreOffice Calc",
    "Google Chrome", "Mozilla Firefox", "Microsoft Edge", "Brave", "Opera",
    "Photoshop", "Illustrator", "Premiere Pro", "GIMP", "Inkscape", "DaVinci Resolve",
    "Docker Desktop", "Postman", "Insomnia", "GitHub Desktop", "Terminal", "PowerShell",
    "Minecraft", "Fortnite", "League of Legends", "Steam", "Battle.net", "Epic Games Launcher",
    "Unity", "Unreal Engine",
]


def simple_similarity(a: str, b: str) -> float:
    a, b = a.lower(), b.lower()
    matches = sum(1 for x in a if x in b)
    return 2.0 * matches / (len(a) + len(b) + 1e-6)


def detect_app_from_text(ocr: OCRResult) -> Optional[Dict[str, Any]]:
    if not ocr.words:
        return None
    haystack = " ".join(ocr.words)
    best_match, best_score = None, 0.0
    for app in KNOWN_APPS:
        score = simple_similarity(haystack, app)
        if score > best_score:
            best_match, best_score = app, score
    if best_match and best_score >= 0.3:
        return {"name": best_match, "score": round(best_score, 2)}
    return None


# ---------------------------
# Clasificador principal
# ---------------------------

@dataclass
class Description:
    type: str
    summary: str
    details: Dict[str, Any]


def describe_image(path: str) -> Description:
    img = load_image(path)
    img = resize_max(img)

    rects = estimate_window_rects(img)
    ocr = run_ocr(img)

    code_info = detect_code_and_language(ocr)
    app_info = detect_app_from_text(ocr)

    if code_info.get("has_code"):
        lang = code_info.get("language")
        lang_part = f" ({lang})" if lang else ""
        summary = f"Se detecta código{lang_part}."
        return Description(
            type="code",
            summary=summary,
            details={
                "language": lang,
                "ocr_text_density": ocr.density,
                "code_evidence": code_info.get("evidence", {}),
            }
        )

    if app_info:
        summary = f"Se detecta una aplicación posiblemente: {app_info['name']} (score {app_info['score']})."
        return Description(
            type="application",
            summary=summary,
            details={
                "app": app_info,
                "ocr_text_density": ocr.density,
                "window_rects": rects,
            }
        )

    if ocr.density < 5 and rects <= 1:
        return Description(
            type="idle",
            summary="No se observa actividad relevante.",
            details={
                "ocr_text_density": ocr.density,
                "window_rects": rects,
            }
        )

    return Description(
        type="unknown",
        summary="No se pudo clasificar con las reglas actuales.",
        details={
            "ocr_text_density": ocr.density,
            "window_rects": rects,
        }
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python descriptor_simple.py ruta/a/captura.png")
        sys.exit(1)
    path = sys.argv[1]
    desc = describe_image(path)
    print(json.dumps(asdict(desc), ensure_ascii=False, indent=2))
