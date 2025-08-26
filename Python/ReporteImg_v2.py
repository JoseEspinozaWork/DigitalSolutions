from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, AutoModelForCausalLM
from deep_translator import GoogleTranslator
import torch, sys, os, textwrap, argparse

MODEL_ID = "microsoft/Florence-2-base"   # puedes probar "microsoft/Florence-2-large" si tu RAM lo permite

def parse_args():
    p = argparse.ArgumentParser(description="Descripción IA en español de imágenes (Florence-2, CPU).")
    p.add_argument("image", help="Ruta de la imagen (png/jpg/jpeg).")
    p.add_argument("--no-translate", action="store_true", help="No traducir al español (dejar salida original).")
    p.add_argument("--out", type=str, default=None, help="Guardar el informe en un .txt (ruta de salida).")
    p.add_argument("--temp", type=float, default=0.8, help="Temperature para generación (creatividad).")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p para muestreo.")
    p.add_argument("--tokens", type=int, default=220, help="Máximo de tokens nuevos por tarea.")
    return p.parse_args()

def safe_translate(txt: str, target="es") -> str:
    txt = (txt or "").strip()
    if not txt:
        return ""
    try:
        return GoogleTranslator(source="auto", target=target).translate(txt) or ""
    except Exception:
        return txt  # fallback: deja el original si falla la traducción

def run_task(processor, model, image, tag: str, max_new_tokens=220, temperature=0.8, top_p=0.9) -> str:
    inputs = processor(text=tag, images=image, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
        )
    result = processor.batch_decode(out, skip_special_tokens=True)[0]
    return (result or "").strip()

def build_report(caption_es: str, dense_es: str, objects_es: str, ocr_es: str) -> str:
    # Limpieza segura
    def norm(s):
        s = (s or "").replace("\n", " ").strip()
        while "  " in s:
            s = s.replace("  ", " ")
        return s

    caption_es = norm(caption_es)
    dense_es   = norm(dense_es)
    objects_es = norm(objects_es)
    ocr_es     = norm(ocr_es)

    reporte = []
    reporte.append("🖼️ Descripción general")
    reporte.append(textwrap.fill(caption_es or "Sin descripción", width=100))

    if dense_es and dense_es.lower() != (caption_es or "").lower():
        reporte.append("\n🔎 Detalles por regiones")
        reporte.append(textwrap.fill(dense_es, width=100))

    if objects_es:
        reporte.append("\n📦 Objetos detectados")
        reporte.append(objects_es)

    if ocr_es:
        reporte.append("\n📝 Texto detectado (OCR)")
        reporte.append(ocr_es)

    if not ocr_es and not objects_es:
        reporte.append("\n💡 Nota")
        reporte.append("No se detectó texto ni objetos específicos; la escena parece más contextual que informativa.")

    return "\n".join(reporte)

def main():
    args = parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"[ERROR] No existe el archivo: {image_path}")
        sys.exit(3)
    if os.path.isdir(image_path):
        print(f"[ERROR] La ruta apunta a un directorio, no a una imagen: {image_path}")
        sys.exit(4)

    # Abre imagen
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        print("[ERROR] El archivo no es una imagen válida o está corrupto.")
        sys.exit(5)
    except Exception as e:
        print(f"[ERROR] No se pudo abrir la imagen: {e}")
        sys.exit(5)

    # Carga modelo/processor (CPU)
    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
        model.to("cpu")
        # Opcional: limitar hilos si ves alto uso de CPU:
        # torch.set_num_threads(4)
    except Exception as e:
        print(f"[ERROR] No se pudo cargar el modelo: {e}")
        sys.exit(6)

    # Ejecuta tareas de Florence-2 (en inglés)
    try:
        cap_en   = run_task(processor, model, image, "<MORE_DETAILED_CAPTION>", args.tokens, args.temp, args.top_p)
        dense_en = run_task(processor, model, image, "<DENSE_REGION_CAPTION>", args.tokens, args.temp, args.top_p)
        obj_en   = run_task(processor, model, image, "<OBJECT_DETECTION>",    args.tokens, args.temp, args.top_p)
        ocr_en   = run_task(processor, model, image, "<OCR>",                 args.tokens, args.temp, args.top_p)
    except Exception as e:
        print(f"[ERROR] Fallo durante la generación: {e}")
        sys.exit(7)

    # Traducción a español (o se deja original si --no-translate)
    if args.no_translate:
        cap_es, dense_es, obj_es, ocr_es = cap_en, dense_en, obj_en, ocr_en
    else:
        cap_es   = safe_translate(cap_en,  "es")
        dense_es = safe_translate(dense_en,"es")
        obj_es   = safe_translate(obj_en,  "es")
        ocr_es   = safe_translate(ocr_en,  "es")

    # Informe final
    informe = build_report(cap_es, dense_es, obj_es, ocr_es)
    marco = "\n" + "="*100 + "\n"
    salida = f"{marco}{informe}{marco}"
    print(salida)

    # Guardar a archivo si se pidió
    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(informe + "\n")
            print(f"[OK] Informe guardado en: {args.out}")
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo guardar el archivo: {e}")

if __name__ == "__main__":
    main()


