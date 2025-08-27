# v1_safe.py
from PIL import Image, UnidentifiedImageError
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
from requests.exceptions import RequestException
import sys, os

MODEL_ID = "Salesforce/blip-image-captioning-base"

def caption_image(image_path: str) -> str:
    # 1) Validaciones del archivo
    if not image_path or not isinstance(image_path, str):
        raise ValueError("La ruta de la imagen es inválida.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No existe el archivo: {image_path}")
    if os.path.isdir(image_path):
        raise IsADirectoryError(f"La ruta apunta a un directorio, no a una imagen: {image_path}")

    # 2) Abrir imagen
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError as e:
        raise ValueError(f"El archivo no es una imagen soportada o está corrupto: {image_path}") from e

    # 3) Cargar modelo y processor (con use_fast desactivado para evitar torchvision)
    try:
        processor = BlipProcessor.from_pretrained(MODEL_ID, use_fast=False)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    except Exception as e:
        # típicamente: problemas de red, permisos, o espacio en disco al descargar
        raise RuntimeError(
            "No se pudo cargar el modelo/processor BLIP. "
            "Revisa tu conexión a internet y el espacio en disco."
        ) from e

    # 4) Generar caption en inglés
    try:
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=60)
        caption_en = processor.decode(output[0], skip_special_tokens=True).strip()
        if not caption_en:
            raise RuntimeError("El modelo devolvió una cadena vacía.")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        raise RuntimeError(f"Fallo al generar el caption: {e}") from e

    # 5) Traducir a español (con fallback en caso de error)
    try:
        caption_es = GoogleTranslator(source="auto", target="es").translate(caption_en)
        caption_es = (caption_es or "").strip()
        return caption_es if caption_es else caption_en  # fallback si viene vacío
    except (RequestException, Exception):
        # Si falla la traducción (sin internet, rate limit, etc.), devolvemos inglés
        return caption_en

def main():
    if len(sys.argv) < 2:
        print("Uso: py -3.11 v1_safe.py <ruta_imagen>")
        sys.exit(2)

    img_path = sys.argv[1]

    try:
        caption = caption_image(img_path)
        print("Descripción:", caption)
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(3)
    except IsADirectoryError as e:
        print(f"[ERROR] {e}")
        sys.exit(4)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(5)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(6)
    except KeyboardInterrupt:
        print("\n[INFO] Cancelado por el usuario.")
        sys.exit(130)
    except Exception as e:
        # Último recurso: captura cualquier error inesperado
        print(f"[ERROR] Inesperado: {e.__class__.__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

