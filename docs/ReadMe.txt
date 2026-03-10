TODOs:
//Ejecutar consultas en vLLM multiples ?    -- Mejorable. Testear librerias.
//Incluir smartdoc_ai en la clasificacion.

// Errores de scrapt en Petete
// FAISS index
Mejorar errores
Formularios
Popular RAG
// Esta subiendo todos los pdfs?
sqlite carrera!
//Referenciar, y diferir la respuesta.


sshpass -p diganDar ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=3 maxim@212.69.86.224

sudo systemctl restart bluebill
rsync -avz --exclude='.venv' --exclude='*.db' --exclude='venv' --exclude='*.pyc' --exclude='pycache' -e "sshpass -p diganDar ssh" /Users/delfi/PycharmProjects/BlueBill maxim@212.69.86.224:/home/maxim/PycharmProjects/
rsync -avz --exclude='.venv' --exclude='*.db' --exclude='venv' --exclude='*.pyc' --exclude='pycache' -e "sshpass -p diganDar ssh" /home/maxim/PycharmProjects/BlueBill maxim@212.69.86.224:/home/maxim/PycharmProjects/

# Del servidor a R11
rsync -avz --exclude='.venv' --exclude='venv' --exclude='*.pyc' --exclude='__pycache__' -e "sshpass -p diganDar ssh" maxim@212.69.86.224:/home/maxim/PycharmProjects/BlueBill/ /mnt/BackUps/vs/BlueBill-20251029/
rsync -avz --exclude='.venv' --exclude='venv' --exclude='*.pyc' --exclude='__pycache__' -e "sshpass -p diganDar ssh" maxim@212.69.86.224:/home/maxim/PycharmProjects/BSG/ /mnt/BackUps/vs/BSG-200198/


tar -czf /mnt/BackUps/vs/BlueBill_backup_$(date +%Y-%m-%d_%H-%M-%S).tar.gz -C /home/maxim/PycharmProjects --exclude='.venv' --exclude='venv' BlueBill &




Datos del correo:

mlucius@bluebill.es
CtygiFHVzUmhvXl

IMAP: imap.ionos.es
Puerto: 993
Seguridad: SSL

SMTP: smtp.ionos.es
Puerto: 587
Seguridad: STARTTLS


Desde aqui arrancamos ...
https://claude.ai/chat/32896f2a-d2a8-49fb-936e-fc13a4f22b2c


  Fiscal (AEAT)

  - GET /fiscal/scraper_health — Health del pool del scraper AEAT.
  - POST /fiscal/restart_scraper — Reiniciar pool del scraper.
  - GET /fiscal/scraper_performance — Métricas y recomendaciones de rendimiento del scraper.
  - POST /fiscal/classify — Clasificación fiscal (fusión AEAT + SmartDoc).
  - POST /fiscal/classify_async — Clasificación asíncrona (devuelve jb_id).
  - GET /fiscal/status/{job_id} — Estado de un job asíncrono.
  - GET /fiscal/result/{job_id} — Resultado de un job asíncrono.
  - WS /fiscal/ws/{job_id} — WebSocket para progreso en tiempo real del job.
  - POST /fiscal/validate — Validación rápida de estructura de factura.
  - POST /fiscal/load_consultas — Carga externa de consultas al clasificador.
  - POST /fiscal/cleanup_jobs — Limpieza manual de jobs expirados.



# vLLM setpu:
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-3-12b-it \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --served-model-name gemma-3-12b-it


curl -X POST http://172.24.250.17:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-12b-it",
    "messages": [
      {"role": "system", "content": "Eres un asistente útil."},
      {"role": "user", "content": "Dime un chiste corto."}
    ],
    "max_tokens": 50,
    "temperature": 0.7,
    "stream": false
  }'


curl -X POST "http://212.69.86.224:8001/fiscal/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "identificacion": {
        "id_interno": "FACT-2024-001234",
        "numero_factura": "F-2024-0156",
        "serie": "A"
    },
    "conceptos": [
        {
            "descripcion": "Servicios consultoría informática desarrollo ERP",
            "cantidad": 40,
            "precio_unitario": 37.5,
            "importe_linea": 1500,
            "tipo_iva": 21,
            "codigo_producto": "SERV-INFO-001",
            "unidad_medida": "horas"
        }
    ],
    "receptor": {
        "nombre": "Empresa Ejemplo SL",
        "cif": "B12345678",
        "actividad_cnae": "6201 - Programación informática",
        "sector": "Tecnología",
        "tipo_empresa": "SL"
    },
    "emisor": {
        "nombre": "TechConsult SL",
        "cif": "A98765432",
        "actividad_cnae": "6202 - Consultoría informática",
        "pais_residencia": "España"
    },
    "importes": {
        "base_imponible": 1500,
        "iva_21": 315,
        "irpf": 225,
        "total_factura": 1815
    },
    "contexto_empresarial": {
        "departamento": "IT",
        "centro_coste": "CC-001",
        "proyecto": "PROJ-ERP-2024",
        "porcentaje_afectacion": 100,
        "uso_empresarial": "Exclusivo",
        "justificacion_gasto": "Implementación sistema ERP para mejorar procesos"
    },
    "fiscal": {
        "regimen_iva": "General",
        "retencion_aplicada": true,
        "tipo_retencion": 15,
        "operacion_intracomunitaria": false
    },
    "relacion_comercial": {
        "tipo_relacion": "Tercero independiente",
        "empresa_vinculada": false,
        "operacion_vinculada": false,
        "proveedor_habitual": true
    },
    "fechas": {
        "emision": "2024-10-15",
        "prestacion_servicio": "2024-10-01",
        "pago_efectivo": "2024-10-20"
    }
}'


#=========================================================================================
# Iniciar clasificación asíncrona
curl -X POST "http://212.69.86.224:8001/fiscal/classify_async" \
  -H "Content-Type: application/json" \
  -d '{
    "identificacion": {
        "id_interno": "FACT-2024-001234",
        "numero_factura": "F-2024-0156",
        "serie": "A"
    },
    "conceptos": [
        {
            "descripcion": "Servicios consultoría informática desarrollo ERP",
            "cantidad": 40,
            "precio_unitario": 37.5,
            "importe_linea": 1500,
            "tipo_iva": 21,
            "codigo_producto": "SERV-INFO-001",
            "unidad_medida": "horas"
        }
    ],
    "receptor": {
        "nombre": "Empresa Ejemplo SL",
        "cif": "B12345678",
        "actividad_cnae": "6201 - Programación informática",
        "sector": "Tecnología",
        "tipo_empresa": "SL"
    },
    "emisor": {
        "nombre": "TechConsult SL",
        "cif": "A98765432",
        "actividad_cnae": "6202 - Consultoría informática",
        "pais_residencia": "España"
    },
    "importes": {
        "base_imponible": 1500,
        "iva_21": 315,
        "irpf": 225,
        "total_factura": 1815
    },
    "contexto_empresarial": {
        "departamento": "IT",
        "centro_coste": "CC-001",
        "proyecto": "PROJ-ERP-2024",
        "porcentaje_afectacion": 100,
        "uso_empresarial": "Exclusivo",
        "justificacion_gasto": "Implementación sistema ERP para mejorar procesos"
    },
    "fiscal": {
        "regimen_iva": "General",
        "retencion_aplicada": true,
        "tipo_retencion": 15,
        "operacion_intracomunitaria": false
    },
    "relacion_comercial": {
        "tipo_relacion": "Tercero independiente",
        "empresa_vinculada": false,
        "operacion_vinculada": false,
        "proveedor_habitual": true
    },
    "fechas": {
        "emision": "2024-10-15",
        "prestacion_servicio": "2024-10-01",
        "pago_efectivo": "2024-10-20"
    }
}'

# Usando el job_id de la respuesta anterior
JOB_ID="a1b2c3d4-e5f6-7890-abcd-ef1234567890"

curl -X GET "http://212.69.86.224:8001/fiscal/status/${JOB_ID}" \
  -H "Accept: application/json"



  # Ver logs completos del servicio
sudo journalctl -u bluebill.service --no-pager -n 100

# Buscar errores específicos de binding/puerto
sudo journalctl -u bluebill.service | grep -i -E "(error|bind|port|8001|uvicorn|address already in use)"

# Ver logs desde el último inicio
sudo journalctl -u bluebill.service --since "2025-08-02 22:32:00"
