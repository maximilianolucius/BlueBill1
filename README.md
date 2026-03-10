# BlueBill

Plataforma de clasificacion fiscal automatizada para Espana (AEAT). Combina web scraping de consultas tributarias, clasificacion de facturas con IA (vLLM, embeddings, FAISS) y gestion documental inteligente (SmartDoc) a traves de una API REST unificada.

## Arquitectura

```
bluebill_app.py                  # API principal FastAPI (punto de entrada)
  |
  +-- fiscal_classifier.py       # Motor de clasificacion fiscal (vLLM + embeddings)
  |     +-- clasificaciones_fiscales_aeat_demo.py   # Datos de codigos G AEAT
  |     +-- utils/generator_model.py                # Adaptador LLM (vLLM/Gemini/LangChain)
  |
  +-- aeat_scraper_singleton.py  # Singleton thread-safe del scraper
  |     +-- aeat_scraper.py      # Scraper Selenium del portal AEAT
  |           +-- utils/generar_keywords_y_fiscal_terms.py  # Generador de keywords fiscales
  |
  +-- smartdoc_ai/smartdoc_backend.py  # Backend de gestion documental
        +-- utils/generator_model.py   # Inferencia vLLM
```

## Modulos principales

### `bluebill_app.py`
Servidor API unificado (FastAPI v4.0). Combina SmartDoc + Clasificacion Fiscal con:
- Persistencia SQLite para documentos y jobs
- Cola de trabajos asincrona con WebSocket para progreso en tiempo real
- Cache AEAT con TTL configurable
- Metricas de rendimiento y rate-limiting

### `fiscal_classifier.py`
Sistema de recomendaciones fiscales usando VLLM + 200+ consultas tributarias oficiales AEAT. Incluye:
- Extraccion robusta de datos de factura (`FacturaDataExtractor`)
- Procesamiento de consultas tributarias con embeddings semanticos
- Motor de recomendaciones con scoring multi-factor (CNAE, keywords, sector)
- Embedder minimo de fallback (`MinimalHashingEmbedder`)

### `aeat_scraper.py`
Web scraper automatizado del portal de consultas tributarias de la AEAT:
- Pool de drivers Selenium para busquedas paralelas
- Cache persistente SQLite (TTL configurable, 3 meses por defecto)
- Generacion dinamica de keywords fiscales via LLM

### `aeat_scraper_singleton.py`
Patron singleton thread-safe para gestionar una instancia global del scraper:
- Auto-reinicio basado en tasa de errores
- Estadisticas de rendimiento (requests, errores, tiempos)
- Wrapper async para integracion con FastAPI

### `smartdoc_ai/smartdoc_backend.py`
Backend de gestion documental inteligente:
- Carga y extraccion de texto (PDF/texto plano)
- Deteccion de duplicados por fingerprint
- Indices FAISS por documento y global (busqueda cruzada)
- Resumenes y Q&A via vLLM

### `utils/generator_model.py`
Capa de abstraccion para modelos LLM:
- `VLLMClient`: cliente vLLM con procesamiento async/batch
- `GeminiLLM`: cliente Google Gemini con streaming
- `LangChainVLLMAdapter`: adaptador unificado con fallback automatico a Gemini

### `utils/generar_keywords_y_fiscal_terms.py`
Generador de keywords y terminos fiscales a partir de datos de factura:
- Generacion via LLM (LangChain adapter)
- Fallback heuristico cuando vLLM no esta disponible

## Endpoints API

### Fiscal (AEAT)
| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| `POST` | `/fiscal/classify` | Clasificacion fiscal (fusion AEAT + SmartDoc) |
| `POST` | `/fiscal/classify_async` | Clasificacion asincrona (devuelve job_id) |
| `GET` | `/fiscal/status/{job_id}` | Estado de un job asincrono |
| `GET` | `/fiscal/result/{job_id}` | Resultado de un job asincrono |
| `WS` | `/fiscal/ws/{job_id}` | WebSocket para progreso en tiempo real |
| `POST` | `/fiscal/validate` | Validacion rapida de estructura de factura |
| `POST` | `/fiscal/load_consultas` | Carga externa de consultas al clasificador |
| `GET` | `/fiscal/scraper_health` | Health del pool del scraper AEAT |
| `POST` | `/fiscal/restart_scraper` | Reiniciar pool del scraper |
| `GET` | `/fiscal/scraper_performance` | Metricas del scraper |
| `POST` | `/fiscal/cleanup_jobs` | Limpieza de jobs expirados |

### SmartDoc
| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| `POST` | `/smartdoc/upload` | Subir documento (PDF/texto) |
| `GET` | `/smartdoc/documents` | Listar documentos |
| `GET` | `/smartdoc/document/{id}` | Detalle de documento |
| `GET` | `/smartdoc/document/{id}/summary` | Resumen del documento |
| `POST` | `/smartdoc/document/{id}/query` | Pregunta sobre un documento |
| `POST` | `/smartdoc/query_all` | Busqueda global multi-documento |

### Sistema
| Metodo | Ruta | Descripcion |
|--------|------|-------------|
| `GET` | `/health` | Estado del sistema |

## Scripts auxiliares

| Archivo | Descripcion |
|---------|-------------|
| `factura_analyzer.py` | Analizador de facturas IT con busquedas AEAT especificas |
| `analizar_json.py` | Inspeccion rapida de estructura de datos JSON |
| `analizar_receptor.py` | Deduccion de receptor por frecuencia en facturas |
| `deducir_receptor_por_gastos.py` | Categorizacion de gastos para inferir actividad |
| `transformar_factura.py` | Transformador de formato de facturas |
| `rebuild_faiss_indexes.py` | Reconstruccion de indices FAISS desde SQLite |
| `fix_faiss_deserialization.py` | Reparacion de formatos FAISS en SQLite |
| `clasificaciones_fiscales_aeat_demo.py` | Loader de clasificaciones G-codes AEAT |
| `data_downloaders/manuales_practicos.py` | Descarga de PDFs de manuales AEAT |

## Tests

| Archivo | Descripcion |
|---------|-------------|
| `tests/test_bluebill_app.py` | Suite pytest completa (mocks vLLM, FAISS, endpoints) |
| `tests/test_aeat_scraper_cache.py` | Test del sistema de cache del scraper |
| `tests/test_vllm_status.py` | Test de conectividad vLLM |
| `tests/test_discount_amount_issue.py` | Clasificacion batch de facturas |
| `tests/run_fiscal_classify.py` | Request unica de clasificacion |
| `tests/run_fiscal_classify_seq.py` | Benchmark secuencial (latencia, percentiles) |
| `tests/run_fiscal_classify_parallel.py` | Stress test paralelo (1000 requests) |
| `tests/run_classify_300_parallel.py` | 30 requests paralelas + polling async |
| `tests/run_classify_async_single.py` | Submit + poll de job async individual |
| `tests/run_classify_async_batch.py` | Batch de 4 rondas x 5 facturas async |
| `tests/run_scraper_payload.py` | Ejecucion directa del scraper con payload |

## Herramientas

| Archivo | Descripcion |
|---------|-------------|
| `tools/job_stats.py` | Analitica de jobs en SQLite (stats, errores, cleanup) |
| `tools/query_vllm.py` | Query de diagnostico a vLLM |

## Instalacion

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Configuracion

Variables de entorno (`.env`):

```env
VLLM_BASE_URL=http://172.24.250.17:8000/v1
VLLM_MODEL=Qwen3-8B-AWQ
VLLM_API_KEY=EMPTY
SMARTDOC_DB_PATH=smartdoc_persistence.db
AEAT_POOL_SIZE=64
SMARTDOC_ENRICHMENT=0
```

## Ejecucion

```bash
# Desarrollo
python bluebill_app.py

# Produccion
uvicorn bluebill_app:app --host 0.0.0.0 --port 8001 --workers 4

# Tests
make test
```

## Estructura del proyecto

```
BlueBill/
  bluebill_app.py                     # API principal FastAPI
  aeat_scraper.py                     # Scraper AEAT con pool y cache
  aeat_scraper_singleton.py           # Singleton del scraper
  fiscal_classifier.py                # Clasificador fiscal IA
  clasificaciones_fiscales_aeat_demo.py  # Loader de G-codes AEAT
  smartdoc_ai/
    smartdoc_backend.py               # Backend documental
    examples.py                       # Ejemplos de uso de la API
  utils/
    generator_model.py                # Abstraccion LLM (vLLM/Gemini/LangChain)
    generar_keywords_y_fiscal_terms.py  # Generador de keywords fiscales
  tests/                              # Tests unitarios y benchmarks
  tools/                              # Herramientas de diagnostico
  data_downloaders/                   # Descargadores de datos AEAT
  data/
    aeat_outputs/                     # Resultados del scraper AEAT (JSON + XLSX)
    fiscal/                           # Datos fiscales de referencia
      clasificaciones_fiscales_aeat.json  # Catalogo G-codes AEAT
      modelos.csv                     # Modelos tributarios
    facturas/                         # Datos de facturas
      fg.json                         # Facturas exportadas de PHPMyAdmin
      facturas_transformadas.json     # Facturas en formato estandar
  docs/                               # Documentacion del proyecto
    [BlueBill] Master Doc.pdf         # Documento maestro
    Especificacion*.txt/docx          # Especificaciones tecnicas
    Refactor reference V*.docx        # Documentacion de refactoring
    Modelos Tributarios.docx          # Referencia de modelos
    ReadMe.txt                        # Notas operativas (legacy)
  input/                              # Datos de entrada (SQL, facturas)
  modelos/                            # Modelos y documentos AEAT descargados
  factura_analyzer.py                 # Analizador de facturas IT
  analizar_json.py                    # Inspeccion de estructura JSON
  analizar_receptor.py                # Analisis de receptores
  deducir_receptor_por_gastos.py      # Deduccion por gastos
  transformar_factura.py              # Transformador de facturas
  rebuild_faiss_indexes.py            # Reconstruccion de indices FAISS
  fix_faiss_deserialization.py        # Reparacion FAISS en SQLite
  bench_webquery.py                   # Benchmark HTTP
  benchmark_apis.py                   # Benchmark de API fiscal
  Makefile                            # Comandos de desarrollo y test
  requirements.txt                    # Dependencias Python
  setup.sh                            # Script de instalacion del servidor
  smartdoc_persistence.db             # Base de datos SQLite principal
```
