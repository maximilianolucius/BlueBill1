#!/bin/bash

# =============================================================================
# SCRIPT DE CONFIGURACIÓN SISTEMA CLASIFICACIÓN GASTOS AEAT
# =============================================================================

set -e  # Salir si hay error

echo "🚀 CONFIGURANDO SISTEMA DE CLASIFICACIÓN DE GASTOS AEAT"
echo "======================================================"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Función para imprimir con colores
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# =============================================================================
# 1. VERIFICAR DEPENDENCIAS DEL SISTEMA
# =============================================================================

echo -e "\n📋 VERIFICANDO DEPENDENCIAS..."

# Verificar Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_status "Python encontrado: $PYTHON_VERSION"
else
    print_error "Python3 no encontrado. Instala Python 3.8 o superior"
    exit 1
fi

# Verificar pip
if command -v pip3 &> /dev/null; then
    print_status "pip3 disponible"
else
    print_error "pip3 no encontrado. Instala pip3"
    exit 1
fi

# Verificar curl
if command -v curl &> /dev/null; then
    print_status "curl disponible"
else
    print_error "curl no encontrado. Instala curl"
    exit 1
fi

# =============================================================================
# 2. VERIFICAR SERVIDOR LLM
# =============================================================================

echo -e "\n🤖 VERIFICANDO SERVIDOR LLM..."

# Verificar si hay un servidor corriendo en puerto 8000
if curl -s http://172.24.250.17:8000/v1/models > /dev/null; then
    print_status "Servidor LLM encontrado en puerto 8000"
else
    print_warning "No se encontró servidor LLM en puerto 8000"
    print_warning "Asegúrate de tener un servidor compatible con API OpenAI ejecutándose"
    print_warning "Ejemplo: servidor HuggingFace TGI con google/gemma-3-27b-it"

    # Continuar sin salir, ya que el usuario puede configurar esto después
fi

# =============================================================================
# 3. VERIFICAR MODELO DISPONIBLE
# =============================================================================

echo -e "\n🧠 VERIFICANDO MODELO GEMMA-3-27B-IT..."

# Probar el modelo
TEST_RESPONSE=$(curl -s -X POST http://172.24.250.17:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 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" \
    -d '{
        "model": "google/gemma-3-27b-it",
        "messages": [{"role": "user", "content": "Responde solo: OK"}],
        "max_tokens": 10
    }' 2>/dev/null)

if echo "$TEST_RESPONSE" | grep -q "choices"; then
    print_status "Modelo google/gemma-3-27b-it responde correctamente"
else
    print_warning "Modelo no responde o no está disponible"
    print_warning "Respuesta del servidor: $TEST_RESPONSE"
fi

# =============================================================================
# 4. INSTALAR DEPENDENCIAS PYTHON
# =============================================================================

echo -e "\n📦 INSTALANDO DEPENDENCIAS PYTHON..."

# Crear requirements.txt si no existe
if [ ! -f "requirements.txt" ]; then
    cat > requirements.txt << EOF
flask==2.3.3
requests==2.31.0
sentence-transformers==2.2.2
numpy==1.24.3
scikit-learn==1.3.0
beautifulsoup4==4.12.2
gunicorn==21.2.0
python-dotenv==1.0.0
werkzeug==2.3.7
lxml==4.9.3
EOF
    print_status "Archivo requirements.txt creado"
fi

# Instalar dependencias
print_warning "Instalando dependencias Python..."
source venv/bin/activate
pip3 install -r requirements.txt

# Instalar LangChain obligatoriamente
pip3 install "langchain>=0.2,<0.3" "langchain-openai>=0.1,<0.2"

if [ $? -eq 0 ]; then
    print_status "Dependencias Python instaladas exitosamente"
else
    print_error "Error instalando dependencias Python"
    exit 1
fi

# =============================================================================
# 5. CONFIGURAR BASE DE DATOS
# =============================================================================

echo -e "\n🗄️  CONFIGURANDO BASE DE DATOS..."

# Crear directorio para datos si no existe
mkdir -p data
mkdir -p logs

print_status "Directorios de datos creados"

# =============================================================================
# 6. PROBAR CONFIGURACIÓN
# =============================================================================

echo -e "\n🧪 PROBANDO CONFIGURACIÓN..."

# Probar Ollama
print_warning "Probando conexión con Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null; then
    print_status "Ollama responde correctamente"
else
    print_error "Ollama no responde en puerto 11434"
    exit 1
fi

# Probar modelo Gemma2
print_warning "Probando modelo Gemma2..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma2:27b",
        "prompt": "Responde solo: OK",
        "stream": false,
        "options": {"max_tokens": 10}
    }')

if echo "$TEST_RESPONSE" | grep -q "response"; then
    print_status "Modelo Gemma2 responde correctamente"
else
    print_error "Modelo Gemma2 no responde correctamente"
    echo "Respuesta: $TEST_RESPONSE"
    exit 1
fi

# =============================================================================
# 7. INICIAR SERVICIO DE CLASIFICACIÓN
# =============================================================================

echo -e "\n🚀 INICIANDO SERVICIO..."

# Verificar si app.py existe
if [ ! -f "app.py" ]; then
    print_error "Archivo app.py no encontrado"
    print_warning "Copia el código del sistema al archivo app.py"
    exit 1
fi

# Iniciar servicio en segundo plano
print_warning "Iniciando servicio de clasificación..."
python3 app.py &
APP_PID=$!

# Esperar a que el servicio esté listo
sleep 10

# Probar servicio
if curl -s http://localhost:5000/api/get_codes > /dev/null; then
    print_status "Servicio de clasificación iniciado correctamente"
    print_status "PID del proceso: $APP_PID"
else
    print_error "Servicio de clasificación no responde"
    kill $APP_PID 2>/dev/null
    exit 1
fi

# =============================================================================
# 8. EJECUTAR PRUEBAS
# =============================================================================

echo -e "\n🧪 EJECUTANDO PRUEBAS..."

# Crear script de prueba simple
cat > test_simple.py << 'EOF'
import requests
import json

def test_clasificacion():
    factura_prueba = {
        "identificacion": {
            "numero_factura": "F-2024-001",
            "serie": "A"
        },
        "emisor": {
            "nombre": "TechConsulting SL",
            "cif": "B12345678",
            "actividad_cnae": "6202"
        },
        "receptor": {
            "nombre": "MiEmpresa SA",
            "actividad_cnae": "4711",
            "tipo_empresa": "PYME"
        },
        "conceptos": [
            {
                "descripcion": "Servicios de consultoría informática",
                "importe_linea": "1500"
            }
        ],
        "importes": {
            "total": "1815"
        },
        "contexto_empresarial": {
            "departamento": "IT"
        }
    }

    try:
        response = requests.post(
            "http://localhost:5000/api/classify_expense",
            json=factura_prueba,
            timeout=30
        )

        if response.status_code == 200:
            resultado = response.json()
            print(f"✅ Clasificación exitosa: {resultado.get('codigo_principal', 'ERROR')}")
            print(f"   Confianza: {resultado.get('confianza', 0):.2f}")
            print(f"   Justificación: {resultado.get('justificacion', 'N/A')[:100]}...")

            # Mostrar precedentes si existen
            if 'precedentes_oficiales' in resultado:
                print(f"   Precedentes: {len(resultado['precedentes_oficiales'])} encontrados")

            # Mostrar oportunidades si existen
            if 'oportunidades_detectadas' in resultado:
                print(f"   Oportunidades: {len(resultado['oportunidades_detectadas'])} detectadas")

            return True
        else:
            print(f"❌ Error HTTP: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Ejecutando prueba de clasificación...")
    if test_clasificacion():
        print("✅ Prueba exitosa")
    else:
        print("❌ Prueba fallida")
EOF

# Ejecutar prueba
python3 test_simple.py
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    print_status "Pruebas completadas exitosamente"
else
    print_error "Las pruebas fallaron"
fi

# =============================================================================
# 9. INFORMACIÓN FINAL
# =============================================================================

echo -e "\n🎉 CONFIGURACIÓN COMPLETADA"
echo "=========================="
echo ""
echo "📊 INFORMACIÓN DEL SISTEMA:"
echo "- Servicio API: http://localhost:5000"
echo "- Servidor LLM: http://172.24.250.17:8000"
echo "- Modelo: google/gemma-3-27b-it"
echo "- Base de datos: expense_learning.db"
echo "- PID servicio: $APP_PID"
echo ""
echo "📝 ENDPOINTS DISPONIBLES:"
echo "- POST /api/classify_expense      - Clasificar gasto (JSON completo)"
echo "- POST /api/correct_classification - Enviar corrección"
echo "- GET  /api/get_codes             - Obtener códigos AEAT"
echo "- GET  /api/stats                 - Estadísticas del sistema"
echo ""
echo "🔧 COMANDOS ÚTILES:"
echo "- Ver logs: tail -f logs/app.log"
echo "- Parar servicio: kill $APP_PID"
echo "- Verificar LLM: curl http://172.24.250.17:8000/v1/models"
echo "- Ejecutar ejemplos: python3 examples.py"
echo ""
echo "📚 PRÓXIMOS PASOS:"
echo "1. Ejecuta: python3 examples.py"
echo "2. Prueba clasificaciones con tus propias facturas"
echo "3. Envía correcciones para mejorar el sistema"
echo "4. Monitorea estadísticas en /api/stats"
echo ""

# Crear archivo de estado
cat > system_status.txt << EOF
Sistema de Clasificación AEAT - Estado
=====================================
Fecha instalación: $(date)
PID servicio: $APP_PID
Puerto API: 5000
Puerto LLM: 8000
Modelo: google/gemma-3-27b-it
Base datos: expense_learning.db
Token HF: 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

Para parar el sistema:
kill $APP_PID

Para verificar LLM:
curl http://172.24.250.17:8000/v1/models
EOF

print_status "Sistema configurado y listo para usar"
print_warning "Archivo de estado guardado en: system_status.txt"

# Mantener el script corriendo para mostrar logs
echo -e "\n📋 MONITOREANDO SERVICIO (Ctrl+C para salir)..."
echo "Logs en tiempo real:"
echo "===================="

# Seguir logs del servicio
tail -f logs/app.log 2>/dev/null || tail -f /dev/null
