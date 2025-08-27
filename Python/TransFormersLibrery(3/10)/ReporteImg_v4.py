# v4_code_inspector_multi.py
# Múltiples imágenes:
#  - Auto-recorte (opcional) de la zona con más texto (--auto-crop)
#  - OCR: Florence-2 (por defecto) o Tesseract (--ocr tesseract)
#  - Limpieza de OCR y detección de lenguaje (Pygments + heurísticas)
#  - Salida en consola por imagen y guardado a archivos por imagen (--out-dir) o combinado (--out)

from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, AutoModelForCausalLM
import torch, sys, os, re, argparse, textwrap

# ---- opcionales (silenciosos) ----
try:
    from deep_translator import GoogleTranslator
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
FLORENCE_PROCESSOR = None
FLORENCE_MODEL = None

def parse_args():
    p = argparse.ArgumentParser(description="Analiza una o varias capturas de código con OCR y detección de lenguaje.")
    p.add_argument("images", nargs="+", help="Ruta(s) de imagen (png/jpg/jpeg).")
    p.add_argument("--ocr", choices=["florence", "tesseract"], default="florence", help="Motor OCR (default: florence).")
    p.add_argument("--no-translate", action="store_true", help="No traducir OCR al español.")
    p.add_argument("--tesseract-path", type=str, default=None, help="Ruta a tesseract.exe si no está en PATH (Windows).")
    p.add_argument("--tokens", type=int, default=160, help="Máx. tokens de Florence para <OCR>.")
    p.add_argument("--temp", type=float, default=0.7, help="Temperature en Florence.")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p en Florence.")
    p.add_argument("--auto-crop", action="store_true", help="Recortar automáticamente la región con más texto (requiere pytesseract).")
    p.add_argument("--save-crop", type=str, default=None, help="Guardar recortes (si carpeta, se guarda uno por imagen).")
    p.add_argument("--out-dir", type=str, default=None, help="Guardar un informe .txt por imagen en esta carpeta.")
    p.add_argument("--out", type=str, default=None, help="Guardar también un informe combinado con todas las imágenes.")
    return p.parse_args()

# ---------------- utilidades de archivos ----------------

def ensure_dir(path: str | None):
    if not path:
        return
    if os.path.isfile(path):
        # Si es archivo, no crear carpeta
        return
    os.makedirs(path, exist_ok=True)

def path_is_dir(path: str | None) -> bool:
    return bool(path) and os.path.isdir(path)

def build_per_image_path(base_dir: str | None, image_path: str, suffix: str) -> str | None:
    if not base_dir:
        return None
    ensure_dir(base_dir)
    name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(base_dir, f"{name}{suffix}")

# --------------- OCR: Florence y Tesseract ----------------

def load_florence():
    global FLORENCE_PROCESSOR, FLORENCE_MODEL
    if FLORENCE_PROCESSOR is None or FLORENCE_MODEL is None:
        FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        FLORENCE_MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to("cpu")

def ocr_with_florence(image: Image.Image, tokens=160, temperature=0.7, top_p=0.9) -> str:
    load_florence()
    inputs = FLORENCE_PROCESSOR(text="<OCR>", images=image, return_tensors="pt")
    with torch.inference_mode():
        out = FLORENCE_MODEL.generate(**inputs, max_new_tokens=tokens, do_sample=True, top_p=top_p, temperature=temperature)
    txt = FLORENCE_PROCESSOR.batch_decode(out, skip_special_tokens=True)[0]
    return (txt or "").strip()

def ocr_with_tesseract(image: Image.Image, lang="eng+spa") -> str:
    txt = pytesseract.image_to_string(image, lang=lang)
    return (txt or "").strip()

# --------------- Auto-crop por densidad de texto ---------------

def auto_crop_text_region(image: Image.Image, lang="eng+spa", conf_threshold=60, pad_ratio=0.03):
    if not HAS_TESSERACT:
        return image, None
    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    n = len(data.get("text", []))
    boxes = []
    for i in range(n):
        txt = data["text"][i]
        conf = data["conf"][i]
        if not txt or not txt.strip() or conf in ("-1", "", None):
            continue
        try:
            c = float(conf)
        except ValueError:
            c = -1.0
        if c >= conf_threshold:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            boxes.append((x, y, w, h))
    if not boxes:
        return image, None
    min_x = min(x for x,_,w,_ in boxes)
    min_y = min(y for _,y,_,_ in boxes)
    max_x = max(x+w for x,_,w,_ in boxes)
    max_y = max(y+h for _,y,_,h in boxes)
    W, H = image.size
    pad = int(pad_ratio * min(W, H))
    x0 = max(min_x - pad, 0); y0 = max(min_y - pad, 0)
    x1 = min(max_x + pad, W); y1 = min(max_y + pad, H)
    area_crop = (x1-x0)*(y1-y0); area_full = W*H
    if area_crop / max(area_full,1) > 0.95:
        return image, None
    return image.crop((x0,y0,x1,y1)), (x0,y0,x1,y1)

# --------------- Limpieza y detección de lenguaje ---------------

def clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\ufeff"," ").replace("\u200b"," ").replace("\u00ad","")
    lines = text.splitlines()
    out = []
    for ln in lines:
        ln2 = re.sub(r"^\s*\d+\s+", "", ln)  # quitar números de línea
        ln2 = ln2.replace("“","\"").replace("”","\"").replace("’","'").replace("‘","'")
        out.append(ln2.rstrip())
    s = "\n".join(out)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

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
    return sum(1 for pat in patterns if re.search(pat, text, flags=re.MULTILINE))

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

def infer_py_version(code: str) -> str:
    if re.search(r"from\s+__future__\s+import\s+print_function", code): return "3"
    has_print_call = re.search(r"\bprint\s*\(", code) is not None
    has_print_stmt = re.search(r"^\s*print\s+[^\(].*$", code, flags=re.MULTILINE) is not None
    if has_print_stmt and not has_print_call: return "2"
    if has_print_call: return "3"
    return "n/a"

def to_spanish(s: str) -> str:
    s = (s or "").strip()
    if not s: return ""
    if not HAS_TRANSLATOR: return s
    try:
        return (GoogleTranslator(source="auto", target="es").translate(s) or "").strip()
    except Exception:
        return s

def describe_paragraph(code_text: str) -> str:
    txt = code_text.strip()
    if not txt: return "Captura sin texto legible."
    single = " ".join(txt.split())
    if len(single) > 600: single = single[:600] + "..."
    return to_spanish(single)

# ---------------------- Reporte por imagen ----------------------

def build_report_for_image(image_name: str, ocr_clean: str, lang: str, conf: int, ranking, version: str, ext: str, crop_box):
    wrap = lambda s: textwrap.fill(s, width=100)
    desc = describe_paragraph(ocr_clean)
    header = f"Imagen: {image_name}\nLenguaje: {lang} (versión {version}, extensión {ext})"
    if crop_box:
        x0,y0,x1,y1 = crop_box
        header += f"\nRecorte auto: x0={x0}, y0={y0}, x1={x1}, y1={y1}"
    body = f"\nDescripción:\n{wrap(desc)}"
    footer = f"\nConfianza aprox.: {conf}% | Top: " + ", ".join([f"{l}:{s}" for l,s in (ranking or [])])
    line = "="*100
    return f"{header}{body}\n{footer}\n{line}\n"

# ----------------------------- Main -----------------------------

def main():
    args = parse_args()

    # tesseract.exe si se indicó
    if args.tesseract_path and HAS_TESSERACT:
        try:
            pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
        except Exception:
            pass

    # preparar directorios
    if args.out_dir: ensure_dir(args.out_dir)
    # --save-crop como carpeta (recomendado)
    save_crops_as_dir = path_is_dir(args.save_crop) or (args.save_crop and not os.path.splitext(args.save_crop)[1])

    combined_reports = []

    for img_path in args.images:
        base = os.path.basename(img_path)

        # validar imagen
        if not os.path.exists(img_path) or os.path.isdir(img_path):
            report = f"Imagen: {base}\nLenguaje: desconocido (versión n/a, extensión n/a)\nDescripción:\nNo existe el archivo o es un directorio.\n{'='*100}\n"
            print(report, end="")
            combined_reports.append(report)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            report = f"Imagen: {base}\nLenguaje: desconocido (versión n/a, extensión n/a)\nDescripción:\nArchivo no es imagen válida.\n{'='*100}\n"
            print(report, end="")
            combined_reports.append(report)
            continue
        except Exception as e:
            report = f"Imagen: {base}\nLenguaje: desconocido (versión n/a, extensión n/a)\nDescripción:\nNo se pudo abrir: {e}\n{'='*100}\n"
            print(report, end="")
            combined_reports.append(report)
            continue

        # auto-crop
        crop_box = None
        if args.auto_crop and HAS_TESSERACT:
            cropped, crop_box = auto_crop_text_region(image, lang="eng+spa", conf_threshold=60, pad_ratio=0.03)
            # guardar recorte
            if args.save_crop:
                if save_crops_as_dir:
                    ensure_dir(args.save_crop)
                    out_crop = build_per_image_path(args.save_crop, img_path, "_crop.png")
                else:
                    # si pasaron un archivo único, generamos nombre a partir de la imagen
                    root, ext0 = os.path.splitext(args.save_crop)
                    out_crop = f"{root}_{os.path.splitext(base)[0]}_crop.png"
                try:
                    cropped.save(out_crop)
                except Exception:
                    pass
            image = cropped

        # OCR
        if args.ocr == "tesseract":
            if not HAS_TESSERACT:
                ocr_raw = ""
            else:
                ocr_raw = ocr_with_tesseract(image, lang="eng+spa")
        else:
            try:
                ocr_raw = ocr_with_florence(image, tokens=args.tokens, temperature=args.temp, top_p=args.top_p)
            except Exception:
                # fallback a Tesseract si está
                if HAS_TESSERACT:
                    try:
                        ocr_raw = ocr_with_tesseract(image, lang="eng+spa")
                    except Exception:
                        ocr_raw = ""
                else:
                    ocr_raw = ""

        # limpiar y detectar
        ocr_clean = clean_ocr_text(ocr_raw)
        lang, conf, ranking = detect_language(ocr_clean)
        ext = {
            "python": ".py","java":".java","csharp":".cs","javascript":".js","typescript":".ts","cpp":".cpp","c":".c",
            "go":".go","rust":".rs","php":".php","ruby":".rb","kotlin":".kt","swift":".swift","sql":".sql","html":".html",
            "css":".css","shell":".sh","powershell":".ps1"
        }.get(lang, "n/a")
        ver = "n/a"
        if lang == "python":
            ver = infer_py_version(ocr_clean)

        # reporte por imagen
        report = build_report_for_image(base, ocr_clean, lang, conf, ranking, ver, ext, crop_box)
        print(report, end="")
        combined_reports.append(report)

        # out por imagen
        if args.out_dir:
            out_txt = build_per_image_path(args.out_dir, img_path, "_report.txt")
            try:
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(report)
            except Exception:
                pass

    # out combinado
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("".join(combined_reports))
            print(f"[OK] Informe combinado guardado en: {args.out}")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo guardar el combinado: {e}")

if __name__ == "__main__":
    main()
