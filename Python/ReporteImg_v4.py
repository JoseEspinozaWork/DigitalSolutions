# code_scan_paragraph.py
# Entrada: una o varias imágenes. Salida: para cada imagen:
# Imagen: <nombre>
# Lenguaje: <lenguaje> (versión, extensión)
# Descripción: <párrafo en español del contenido OCR>
# ============================================================

from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, AutoModelForCausalLM
import torch, sys, os, re, textwrap

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

LANG_EXT = {
    "python": ".py", "java": ".java", "csharp": ".cs", "javascript": ".js",
    "typescript": ".ts", "cpp": ".cpp", "c": ".c", "go": ".go", "rust": ".rs",
    "php": ".php", "ruby": ".rb", "kotlin": ".kt", "swift": ".swift",
    "sql": ".sql", "html": ".html", "css": ".css", "shell": ".sh", "powershell": ".ps1"
}

LANG_PATTERNS = {
    "python":  [r"\bdef\b", r"\bimport\b", r"\bself\b", r":\s*$", r"^\s*#"],
    "java":    [r"\bpublic\b", r"\bclass\b", r"\bstatic\b", r"\bvoid\b", r";\s*$"],
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

def load_florence():
    global FLORENCE_PROCESSOR, FLORENCE_MODEL
    if FLORENCE_PROCESSOR is None or FLORENCE_MODEL is None:
        FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        FLORENCE_MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to("cpu")

def ocr_florence(image: Image.Image) -> str:
    load_florence()
    inputs = FLORENCE_PROCESSOR(text="<OCR>", images=image, return_tensors="pt")
    with torch.inference_mode():
        out = FLORENCE_MODEL.generate(**inputs, max_new_tokens=180, do_sample=True, top_p=0.9, temperature=0.7)
    txt = FLORENCE_PROCESSOR.batch_decode(out, skip_special_tokens=True)[0]
    return (txt or "").strip()

def ocr_tesseract(image: Image.Image) -> str:
    txt = pytesseract.image_to_string(image, lang="eng+spa")
    return (txt or "").strip()

def detect_text_crop(image: Image.Image):
    if not HAS_TESSERACT:
        return image
    try:
        data = pytesseract.image_to_data(image, lang="eng+spa", output_type=Output.DICT)
        n = len(data.get("text", []))
        boxes = []
        for i in range(n):
            txt = data["text"][i]
            conf = data["conf"][i]
            if not txt or not txt.strip(): continue
            try: c = float(conf)
            except: c = -1.0
            if c >= 60:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                boxes.append((x, y, w, h))
        if not boxes: return image
        min_x = min(x for x,_,w,_ in boxes)
        min_y = min(y for _,y,_,_ in boxes)
        max_x = max(x+w for x,_,w,_ in boxes)
        max_y = max(y+h for _,y,_,h in boxes)
        W,H = image.size
        pad = int(0.03*min(W,H))
        x0,y0 = max(min_x-pad,0), max(min_y-pad,0)
        x1,y1 = min(max_x+pad,W), min(max_y+pad,H)
        area_crop=(x1-x0)*(y1-y0); area_full=W*H
        if area_crop/max(area_full,1)>0.95: return image
        return image.crop((x0,y0,x1,y1))
    except Exception:
        return image

def clean_text(text: str) -> str:
    if not text: return ""
    text = text.replace("\ufeff"," ").replace("\u200b"," ").replace("\u00ad","")
    lines = text.splitlines()
    out=[]
    for ln in lines:
        ln2=re.sub(r"^\s*\d+\s+","",ln)
        ln2=ln2.replace("“","\"").replace("”","\"").replace("’","'").replace("‘","'")
        out.append(ln2.rstrip())
    s="\n".join(out)
    s=re.sub(r"\n{3,}","\n\n",s)
    return s.strip()

def guess_language(code: str):
    cleaned=code.strip()
    ranking={}
    try:
        lex=guess_lexer(cleaned); name=(lex.name or "").lower(); ranking[name]=10
    except ClassNotFound: pass
    except: pass
    for lang,pats in LANG_PATTERNS.items():
        score=sum(1 for pat in pats if re.search(pat,cleaned,flags=re.MULTILINE))
        if score>0: ranking[lang]=ranking.get(lang,0)+score
    ordered=sorted(ranking.items(),key=lambda x:x[1],reverse=True)
    best=ordered[0][0] if ordered else "desconocido"
    return best

def infer_py_version(code:str)->str:
    if re.search(r"from\s+__future__\s+import\s+print_function",code): return "3"
    has_print_call=re.search(r"\bprint\s*\(",code) is not None
    has_print_stmt=re.search(r"^\s*print\s+[^\(].*$",code,flags=re.MULTILINE) is not None
    if has_print_stmt and not has_print_call: return "2"
    if has_print_call: return "3"
    return "n/a"

def to_spanish(s:str)->str:
    s=(s or "").strip()
    if not s: return ""
    if not HAS_TRANSLATOR: return s
    try:
        return (GoogleTranslator(source="auto", target="es").translate(s) or "").strip()
    except: return s

def describe_paragraph(code_text:str)->str:
    txt=code_text.strip()
    if not txt: return "Captura sin texto legible."
    single=" ".join(txt.split())
    if len(single)>500: single=single[:500]+"..."
    return to_spanish(single)

def process_image(path:str):
    base=os.path.basename(path)
    try: img=Image.open(path).convert("RGB")
    except UnidentifiedImageError:
        print(f"Imagen: {base}\nLenguaje: desconocido\nDescripción:\nArchivo no es imagen válida.\n{'='*60}")
        return
    except Exception as e:
        print(f"Imagen: {base}\nLenguaje: desconocido\nDescripción:\nNo se pudo abrir: {e}\n{'='*60}")
        return
    img2=detect_text_crop(img)
    ocr_txt=""
    try: ocr_txt=ocr_florence(img2)
    except:
        if HAS_TESSERACT:
            try: ocr_txt=ocr_tesseract(img2)
            except: ocr_txt=""
    cleaned=clean_text(ocr_txt)
    lang=guess_language(cleaned); ext=LANG_EXT.get(lang,"n/a")
    ver="n/a"
    if lang=="python": ver=infer_py_version(cleaned)
    desc=describe_paragraph(cleaned)
    print(f"Imagen: {base}\nLenguaje: {lang} (versión {ver}, extensión {ext})\nDescripción:\n{textwrap.fill(desc,width=100)}\n{'='*60}")

def main():
    if len(sys.argv)<2:
        print("Uso: py -3.11 code_scan_paragraph.py <img1> [img2...]")
        return
    for p in sys.argv[1:]:
        process_image(p)

if __name__=="__main__":
    main()
