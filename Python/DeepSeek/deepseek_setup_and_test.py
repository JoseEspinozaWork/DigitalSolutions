import os, sys, subprocess, textwrap, venv, shutil

VENV_ROOT = r"C:\venvs"
VENV_DIR  = os.path.join(VENV_ROOT, "deepseek")
TMP_DIR   = os.path.join(VENV_DIR, "temp")
PY_EXE    = sys.executable  # python que ejecuta este script

def run(cmd, env=None):
    print(f"\n[cmd] {' '.join(cmd)}")
    subprocess.check_call(cmd, env=env)

def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def create_venv():
    print(f"[*] Creando venv en: {VENV_DIR}")
    builder = venv.EnvBuilder(with_pip=True, clear=False, upgrade=False)
    builder.create(VENV_DIR)

def venv_python():
    # Rutas Windows
    return os.path.join(VENV_DIR, "Scripts", "python.exe")

def venv_pip():
    return os.path.join(VENV_DIR, "Scripts", "pip.exe")

def write_test_script():
    code = textwrap.dedent(r'''
        from deepseek_vl.models import VLChatProcessor
        from transformers import AutoModelForCausalLM
        from deepseek_vl.utils.io import load_pil_images
        import sys, os

        MODEL = "deepseek-ai/deepseek-vl-1.3b-chat"
        print(f"[*] Cargando modelo {MODEL} (la primera vez puede tardar).")
        proc = VLChatProcessor.from_pretrained(MODEL)
        tok = proc.tokenizer
        model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True).eval()

        def describe(path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"No existe la imagen: {path}")
            convo = [{
                "role": "User",
                "content": "<image_placeholder>Describe la captura. Necesito: "
                           "1) ¿Es juego, app o escritorio inactivo? "
                           "2) Nombre del programa/juego si se reconoce. "
                           "3) Si hay código, ¿en qué lenguaje?",
                "images": [path]
            }, {"role": "Assistant", "content": ""}]
            pil_images = load_pil_images(convo)
            inputs = proc(conversations=convo, images=pil_images, force_batchify=True)
            # La implementación del modelo provee prepare_inputs_for_generation (trust_remote_code=True)
            out = model.generate(
                **model.prepare_inputs_for_generation(**inputs),
                max_new_tokens=256,
                do_sample=False
            )
            text = tok.decode(out[0], skip_special_tokens=True)
            return text

        if __name__ == "__main__":
            if len(sys.argv) < 2:
                print("Uso: python test_ds.py C:\\ruta\\a\\screenshot.png")
                sys.exit(1)
            img = sys.argv[1]
            print("[*] Analizando imagen...")
            res = describe(img)
            print("\n=== RESULTADO ===\n")
            print(res)
    ''').strip()
    dst = os.path.join(VENV_DIR, "test_ds.py")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[*] Escrito: {dst}")
    return dst

def main():
    # 1) Carpetas
    ensure_dir(VENV_ROOT)
    # Borra restos de instalaciones a medio hacer si quieres forzar limpio:
    ensure_dir(VENV_DIR)
    ensure_dir(TMP_DIR)

    # 2) Crear venv si falta
    if not os.path.exists(venv_python()):
        create_venv()
    else:
        print("[*] venv ya existe.")

    # 3) Preparar env para que pip use TMP/TEMP dentro del venv (ahorra espacio en C:\Users\...)
    env = os.environ.copy()
    env["TMP"]  = TMP_DIR
    env["TEMP"] = TMP_DIR

    # 4) Actualizar pip del venv
    run([venv_python(), "-m", "pip", "install", "--upgrade", "pip"], env=env)

    # 5) Instalar PyTorch CPU (ligero) + dependencias
    run([venv_pip(), "install", "--no-cache-dir",
         "--index-url", "https://download.pytorch.org/whl/cpu", "torch"], env=env)

    run([venv_pip(), "install", "--no-cache-dir",
         "transformers", "accelerate", "timm",
         "sentencepiece", "attrdict", "einops", "pillow"], env=env)

    # 6) Instalar DeepSeek-VL
    run([venv_pip(), "install", "--no-cache-dir",
         "git+https://github.com/deepseek-ai/DeepSeek-VL.git"], env=env)

    # 7) Crear script de prueba
    test_path = write_test_script()

    print("\n[✓] Instalación completa.")
    print(f"\nPara probar manualmente luego:")
    print(f"  \"{venv_python()}\" \"{test_path}\" \"C:\\ruta\\a\\screenshot.png\"")

    # Si el usuario pasó una imagen como argumento, ejecuta la prueba ahora
    if len(sys.argv) >= 2:
        img = sys.argv[1]
        if os.path.isfile(img):
            print(f"\n[*] Ejecutando prueba automática con: {img}")
            run([venv_python(), test_path, img], env=env)
        else:
            print(f"[!] La ruta de imagen no existe: {img}")

if __name__ == "__main__":
    try:
        main()
        print("\n[FIN]")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Comando falló con código {e.returncode}. Revisa el log arriba.")
        sys.exit(e.returncode)
    except Exception as ex:
        print(f"\n[ERROR] {ex}")
        sys.exit(1)
