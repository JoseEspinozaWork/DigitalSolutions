# v4_code_inspector.py
# Analiza screenshots de código:
#  - (Nuevo) Auto-recorte de la región con más texto (--auto-crop, --save-crop)
#  - OCR con Florence-2 (por defecto) o Tesseract
#  - Limpieza (números de línea, comillas raras)
#  - Detección de lenguaje (Pygments + heurísticas)
#  - Informe y guardado opcional

from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, AutoModelForCausalLM
import torch, sys, os, re, argparse, textwrap

try:
    from deep_translator import GoogleTranslator  # opcional
    HAS_TRANSLATOR = True
except Exception:
    HAS_TRANSLATOR = False

try:
    import pytesseract
    from pytesseract import Output
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

MODEL_ID = "microsoft/Florence-2-base"

def parse_args():
    p = argparse.ArgumentParser(description="Analiza screenshots de código con OCR y detección de lenguaje.")
    p.add_argument("image", help="Ruta de la imagen (png/jpg/jpeg).")
    p.add_argument("--ocr", choices=["florence", "tesseract"], default="florence",
                   help="Motor OCR a usar (por defecto: florence).")
    p.add_argument("--no-translate", action="store_true",
                   help="No traducir el OCR al español.")
    p.add_argument("--tesseract-path", type=str, default=None,
                   help="Ruta al ejecutable de Tesseract si no está en PATH.")
    p.add_argument("--tokens", type=int, default=160,
                   help="Máx. tokens en Florence para <OCR>.")
    p.add_argument("--temp", type=float, default=0.7,
                   help="Creatividad en Florence (no crítico para OCR).")
    p.add_argument("--top_p", type=float, default=0.9,
                   help="Top-p en Florence.")
    p.add_argument("--auto-crop", action="store_true",
                   help="Detectar y recortar automáticamente la región con más texto (requiere pytesseract).")
    p.add_argument("--save-crop", type=str, default=None,
                   help="Guardar la imagen recortada en un archivo.")
    p.add_argument("--out", type=str, default=None,
                   help="Guardar informe en un .txt.")
    return p.parse_args()

# --------------------------- OCR --------------------------------

def set_tesseract_path(path: str | None):
    if path:
        pytesseract.pytesseract.tesseract_cmd = path

def ocr_with_florence(image: Image.Image, tokens=160, temperature=0.7, top_p=0.9) -> str:
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to("cpu")
    inputs = processor(text="<OCR>", images=image, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=tokens, do_sample=True, top_p=top_p, temperature=temperature)
    txt = processor.batch_decode(out, skip_special_tokens=True)[0]
    return (txt or "").strip()

def ocr_with_tesseract(image: Image.Image, lang="eng+spa") -> str:
    txt = pytesseract.image_to_string(image, lang=lang)
    return (txt or "").strip()

# ---------------------- Auto-crop por densidad de texto ----------

def auto_crop_text_region(image: Image.Image, lang="eng+spa", conf_threshold=60, pad_ratio=0.03):
    """
    Usa pytesseract.image_to_data para encontrar cajas de palabras con conf >= conf_threshold.
    Calcula el bbox que cubre la mayoría de palabras y recorta con un padding relativo.
    Si no detecta nada, devuelve la imagen original.
    """
    if not HAS_TESSERACT:
        return image, None  # sin recorte

    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    n = len(data.get("text", []))
    boxes = []
    for i in range(n):
        txt = data["text"][i]
        conf = data["conf"][i]
        if txt and txt.strip() and conf not in ("-1", "", None):
            try:
                c = float(conf)
            except ValueError:
                c = -1.0
            if c >= conf_threshold:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                boxes.append((x, y, w, h))

    if not boxes:
        return image, None

    # Rectángulo envolvente de todas las palabras "confiables"
    min_x = min(x for x,_,w,_ in boxes)
    min_y = min(y for _,y,_,_ in boxes)
    max_x = max(x+w for x,_,w,_ in boxes)
    max_y = max(y+h for _,y,_,h in boxes)

    W, H = image.size
    pad = int(pad_ratio * min(W, H))
    x0 = max(min_x - pad, 0)
    y0 = max(min_y - pad, 0)
    x1 = min(max_x + pad, W)
    y1 = min(max_y + pad, H)

    # Evitar recortes ridículos (p.ej., toda la pantalla)
    # Si el área recortada es >90% del área total, mejor no recortar
    area_crop = (x1 - x0) * (y1 - y0)
    area_full = W * H
    if area_crop / max(area_full, 1) > 0.9:
        return image, None

    cropped = image.crop((x0, y0, x1, y1))
    return cropped, (x0, y0, x1, y1)

# ---------------------- Limpieza y normalización ----------------

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\ufeff", " ").replace("\u200b", " ")
    lines = text.splitlines()
    cleaned = []
    for ln in lines:
        ln2 = re.sub(r"^\s*\d+\s+", "", ln)  # números de línea al inicio
        ln2 = ln2.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
        ln2 = ln2.replace("\u00ad", "")  # soft hyphen
        cleaned.append(ln2.rstrip())
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# --------------------- Detección de lenguaje --------------------

from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

LANG_KEYWORDS = {
    "python":  [r"\bdef\b", r"\bimport\b", r"\bself\b", r":\s*$", r"^\s*#"],
    "java":    [r"\bpublic\b", r"\bclass\b", r"\bstatic\b", r"\bvoid\b", r";\s*$", r"\bSystem\.out\.print"],
    "csharp":  [r"\busing\s+System", r"\bnamespace\b", r"\bclass\b", r"\bConsole\.WriteLine", r";\s*$"],
    "javascript": [r"\bfunction\b", r"\bconst\b", r"\blet\b", r"\bconsole\.log", r";\s*$"],
    "typescript": [r"\binterface\b", r"\btype\b", r":\s*\w+", r"\bimplements\b"],
    "cpp":     [r"#include\s*<", r"\bstd::", r"\bcout\s*<<", r";\s*$"],
    "c":       [r"#include\s*<", r"\bprintf\s*\(", r";\s*$"],
    "go":      [r"\bpackage\s+\w+", r"\bfunc\b", r"\bfmt\.", r":=\s"],
    "rust":    [r"\blet\s+mut\b", r"\bfn\s+\w+\(", r"::", r"println!"],
    "php":     [r"<\?php", r"\becho\b", r"\$\w+", r";\s*$"],
    "ruby":    [r"\bdef\b", r"\bend\b", r":\s*symbol", r"^\s*#"],
    "kotlin":  [r"\bfun\b", r"\bval\b", r"\bvar\b", r"\bdata\s+class\b"],
    "swift":   [r"\bimport\s+Swift", r"\blet\b", r"\bvar\b", r"\bfunc\b"],
    "sql":     [r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b", r"\bJOIN\b", r";\s*$"],
    "html":    [r"<!DOCTYPE html>", r"<\s*html\b", r"<\s*div\b", r"<\s*script\b", r"</\s*\w+\s*>"],
    "css":     [r"\.\w+\s*{", r"#\w+\s*{", r"[a-z-]+\s*:\s*[^;]+;"],
    "shell":   [r"^#!/bin/(ba)?sh", r"\becho\b", r"\bgrep\b", r"\bawk\b"],
    "powershell": [r"Write-Host", r"Get-ChildItem", r"^#\s*Requires", r"\$env:"],
}

def keyword_score(text: str, patterns: list[str]) -> int:
    score = 0
    for pat in patterns:
        if re.search(pat, text, flags=re.MULTILINE):
            score += 1
    return score

def detect_language(code_text: str):
    cleaned = code_text.strip()
    ranking = {}
    try:
        lex = guess_lexer(cleaned)
        name = (lex.name or "").lower()
        ranking[name] = 10
    except ClassNotFound:
        pass
    except Exception:
        pass

    for lang, pats in LANG_KEYWORDS.items():
        ranking[lang] = ranking.get(lang, 0) + keyword_score(cleaned, pats)

    ordered = sorted(ranking.items(), key=lambda x: x[1], reverse=True)
    best, best_score = (ordered[0] if ordered else ("desconocido", 0))
    total = sum(v for _, v in ordered) or 1
    confidence = int(100 * (best_score / total))
    top_list = ordered[:5]
    return best, confidence, top_list

# --------------------------- Informe ----------------------------

def build_report(ocr_text: str, ocr_text_es: str | None, lang_guess: str, conf: int, ranking: list[tuple[str,int]], crop_box):
    wrap = lambda s: textwrap.fill(s, width=100)
    lines = []

    if crop_box:
        x0, y0, x1, y1 = crop_box
        lines.append(f"📐 Recorte auto: x0={x0}, y0={y0}, x1={x1}, y1={y1}")

    lines.append("\n🖹 OCR (texto detectado)")
    lines.append(wrap(ocr_text if ocr_text else "(vacío)"))

    if ocr_text_es is not None:
        lines.append("\n🌐 OCR traducido al español")
        lines.append(wrap(ocr_text_es if ocr_text_es else "(vacío)"))

    lines.append("\n🧠 Detección de lenguaje")
    lines.append(f"Predicción: **{lang_guess}**  |  Confianza aprox.: {conf}%")
    if ranking:
        lines.append("Top candidatos:")
        for lang, score in ranking:
            lines.append(f"  - {lang}: {score}")

    if conf < 50:
        lines.append("\n💡 Nota: La confianza es baja. Intenta un recorte manual o usa --auto-crop.")
    return "\n".join(lines)

# ------------------------------ Main ----------------------------

def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"[ERROR] No existe el archivo: {args.image}")
        sys.exit(2)
    if os.path.isdir(args.image):
        print(f"[ERROR] La ruta apunta a un directorio, no a una imagen: {args.image}")
        sys.exit(3)

    try:
        image = Image.open(args.image).convert("RGB")
    except UnidentifiedImageError:
        print("[ERROR] El archivo no es una imagen válida o está corrupto.")
        sys.exit(4)

    # Tesseract path si se requiere
    if args.tesseract_path and HAS_TESSERACT:
        try:
            pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
        except Exception:
            pass

    # 1) Auto-crop si se pidió y hay pytesseract
    crop_box = None
    if args.auto_crop and HAS_TESSERACT:
        cropped, crop_box = auto_crop_text_region(image, lang="eng+spa", conf_threshold=60, pad_ratio=0.03)
        if args.save_crop and cropped is not None:
            try:
                cropped.save(args.save_crop)
                print(f"[OK] Recorte guardado en: {args.save_crop}")
            except Exception as e:
                print(f"[ADVERTENCIA] No se pudo guardar el recorte: {e}")
        image = cropped

    # 2) OCR
    if args.ocr == "tesseract":
        if not HAS_TESSERACT:
            print("[ERROR] pytesseract no está instalado o no fue importado. Instala 'pytesseract' y Tesseract OCR.")
            sys.exit(5)
        ocr_raw = ocr_with_tesseract(image, lang="eng+spa")
    else:
        ocr_raw = ocr_with_florence(image, tokens=args.tokens, temperature=args.temp, top_p=args.top_p)

    # 3) Limpieza
    ocr_clean = clean_ocr_text(ocr_raw)

    # 4) (Opcional) traducir OCR al español
    if args.no_translate or not HAS_TRANSLATOR:
        ocr_es = None
    else:
        try:
            ocr_es = GoogleTranslator(source="auto", target="es").translate(ocr_clean) if ocr_clean else ""
        except Exception:
            ocr_es = None

    # 5) Detección de lenguaje
    lang, conf, ranking = detect_language(ocr_clean)

    # 6) Informe
    report = build_report(ocr_clean, ocr_es, lang, conf, ranking, crop_box)
    marco = "\n" + "="*100 + "\n"
    print(marco + report + marco)

    # 7) Guardar si se solicitó
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(report + "\n")
            print(f"[OK] Informe guardado en: {args.out}")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo guardar el archivo: {e}")

if __name__ == "__main__":
    main()
