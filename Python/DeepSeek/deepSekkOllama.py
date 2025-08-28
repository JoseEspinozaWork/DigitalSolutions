import base64
import requests
import os
import sys
import argparse

class DeepSeekAnalyzer:
    def __init__(self, ollama_url="http://localhost:11434", model="deepseek-vl2"):
        self.ollama_url = ollama_url
        self.model = model
        self.verify_connection()
    
    def verify_connection(self):
        """Verifica que Ollama esté corriendo y que el modelo existe"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                print("✅ Conexión con Ollama establecida")
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                print(f"📦 Modelos disponibles: {', '.join(available_models)}")
                
                # Buscar algún modelo de DeepSeek
                deepseek_models = [m for m in available_models if 'deepseek' in m.lower()]
                
                if deepseek_models:
                    self.model = deepseek_models[0]
                    print(f"🔍 Usando modelo DeepSeek: {self.model}")
                else:
                    print("❌ No se encontraron modelos DeepSeek")
                    print("📥 Ejecuta: ollama pull deepseek-vl2")
                    sys.exit(1)
                    
            else:
                print("❌ Ollama no responde correctamente")
                sys.exit(1)
                
        except requests.exceptions.ConnectionError:
            print("❌ No se puede conectar a Ollama. Asegúrate de que esté ejecutándose.")
            print("   Ejecuta en terminal: ollama serve")
            sys.exit(1)
    
    def encode_image_to_base64(self, image_path):
        """Convierte imagen a base64"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    def analyze_image(self, image_path, prompt=None):
        """Analiza imagen usando DeepSeek"""
        if prompt is None:
            prompt = """Analiza esta captura de pantalla técnica y describe en español:

1. TIPO: ¿Es código, IDE, editor, juego, aplicación ofimática o otra cosa?
2. LENGUAJE: Si es código, ¿qué lenguaje de programación se visualiza?
3. ENTORNO: ¿Qué software o entorno se está usando? (VS Code, IntelliJ, Eclipse, etc.)
4. ACTIVIDAD: ¿Qué acción está realizando el usuario?
5. DETALLES: Elementos técnicos notables, errores, archivos, estructura.

Responde en un párrafo claro y conciso en español."""
        
        print(f"🔄 Analizando imagen: {os.path.basename(image_path)}")
        print(f"🤖 Usando modelo: {self.model}")
        
        # Codificar imagen
        img_b64 = self.encode_image_to_base64(image_path)
        
        # Preparar payload para DeepSeek (formato específico)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }
        
        try:
            print("⏳ Procesando con DeepSeek... (puede tomar unos segundos)")
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # 2 minutos de timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result['response']
            
        except requests.exceptions.Timeout:
            return "❌ Timeout: El análisis tardó demasiado"
        except requests.exceptions.RequestException as e:
            return f"❌ Error de conexión: {str(e)}"
        except Exception as e:
            return f"❌ Error inesperado: {str(e)}"

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Analizar imágenes con DeepSeek Vision")
    parser.add_argument("image_path", help="Ruta de la imagen a analizar")
    parser.add_argument("-m", "--model", default="deepseek-vl2", help="Modelo DeepSeek a usar")
    parser.add_argument("-p", "--prompt", help="Prompt personalizado")
    
    args = parser.parse_args()
    
    # Verificar que la imagen existe
    if not os.path.exists(args.image_path):
        print(f"❌ Error: La imagen '{args.image_path}' no existe")
        sys.exit(1)
    
    # Inicializar analizador
    analyzer = DeepSeekAnalyzer(model=args.model)
    
    # Analizar imagen
    resultado = analyzer.analyze_image(args.image_path, args.prompt)
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("🔍 RESULTADO DEL ANÁLISIS (DeepSeek):")
    print("="*60)
    print(resultado)
    print("="*60)

if __name__ == "__main__":
    main()