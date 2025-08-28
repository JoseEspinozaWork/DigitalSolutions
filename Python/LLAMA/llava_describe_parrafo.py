import os, sys, base64, ollama

MODEL = 'llava:13b'

SYSTEM_PROMPT = (
    "Eres un analista experto de capturas de pantalla de PC. "
    "Responde SIEMPRE en español en un solo párrafo, sin listas ni JSON. "
    "Prioriza exactitud y evita suposiciones sin evidencia."
)

USER_PROMPT = """Analiza la captura y descríbela en un párrafo.
Indica si muestra código (y lenguaje + IDE), juego (cuál), app de oficina (Word/Excel/PowerPoint),
navegador (Chrome/Edge/Firefox y sitio si lo reconoces), otra app o escritorio inactivo,
y qué parece estar haciendo el usuario. Un solo párrafo, sin listas ni JSON.
"""

def b64(p): return base64.b64encode(open(p,'rb').read()).decode('utf-8')

if len(sys.argv) < 2:
    print("Uso: python describe.py <ruta_imagen>")
    sys.exit(2)

img_b64 = b64(sys.argv[1])

resp = ollama.chat(
    model=MODEL,
    messages=[
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content": USER_PROMPT, "images":[img_b64]}
    ],
    options={
        "temperature": 0.1,
        "num_ctx": 4096,
        "num_predict": 512
    }
)
print("\n--- DESCRIPCIÓN ---\n")
print((resp.get("message") or {}).get("content","").strip() or "[VACÍO]")
