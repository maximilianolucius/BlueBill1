#!/usr/bin/env python3
"""
Sistema de Recomendaciones Fiscales AEAT v2.0.

Módulo principal del clasificador fiscal que combina un modelo de lenguaje
(VLLM) con más de 200 consultas tributarias vinculantes oficiales de la AEAT
para generar recomendaciones de clasificación de gastos empresariales según
los códigos AEAT (G01-G46).

Flujo general:
    1. Se cargan y procesan las consultas tributarias oficiales, generando
       embeddings semánticos para cada una.
    2. Para cada factura de entrada se buscan las consultas más relevantes
       mediante similitud coseno + boosting contextual.
    3. Se construye un prompt enriquecido con los precedentes oficiales y se
       invoca al modelo de lenguaje para obtener la clasificación, análisis
       de deducibilidad, oportunidades fiscales y alertas de cumplimiento.

Clases principales:
    - FiscalPattern: Estructura de datos para patrones fiscales extraídos.
    - FiscalRecommendation: Estructura de datos para recomendaciones fiscales.
    - MinimalHashingEmbedder: Embedder de respaldo basado en hashing.
    - FacturaDataExtractor: Extractor robusto de campos de factura.
    - ConsultasTributariasProcessor: Procesador de consultas tributarias con VLLM.
    - ConsultasRecommendationEngine: Motor de recomendaciones basado en precedentes.
    - EnhancedFiscalClassifier: Orquestador principal del sistema de clasificación.

Dependencias externas:
    - sentence_transformers (SentenceTransformer): embeddings semánticos.
    - numpy: operaciones vectoriales.
    - utils.generator_model.LangChainVLLMAdapter: adaptador del modelo de lenguaje.
    - clasificaciones_fiscales_aeat_demo.ClasificacionesFiscalesAEAT: catálogo de
      clasificaciones fiscales con modelos de presentación.
"""

import json
import time
import logging
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.generator_model import LangChainVLLMAdapter
import sqlite3
from dataclasses import dataclass
from clasificaciones_fiscales_aeat_demo import ClasificacionesFiscalesAEAT

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FiscalPattern:
    """Estructura de datos para patrones fiscales extraídos de consultas tributarias.

    Agrupa la información fiscal relevante detectada mediante análisis de texto
    (VLLM o fallback) sobre una consulta vinculante de la AEAT.

    Atributos:
        codigos_aeat: Códigos AEAT aplicables (p.ej. ``['G19', 'G03']``).
        criterios_deducibilidad: Criterios que determinan si el gasto es deducible.
        incentivos_detectados: Incentivos fiscales detectados (I+D+i, exenciones, etc.).
        cnaes_relevantes: Códigos CNAE mencionados en la consulta.
        condiciones_especiales: Condiciones particulares de aplicación.
        alertas: Alertas o advertencias importantes para el contribuyente.
        keywords_fiscales: Palabras clave fiscales extraídas del texto.
    """
    codigos_aeat: List[str]
    criterios_deducibilidad: List[str]
    incentivos_detectados: List[str]
    cnaes_relevantes: List[str]
    condiciones_especiales: List[str]
    alertas: List[str]
    keywords_fiscales: List[str]


@dataclass
class FiscalRecommendation:
    """Estructura de datos para una recomendación fiscal generada por el sistema.

    Contiene la clasificación del gasto, su justificación, análisis de
    deducibilidad, oportunidades fiscales y alertas de cumplimiento asociadas
    a una factura concreta.

    Atributos:
        codigo_principal: Código AEAT principal recomendado (p.ej. ``'G19'``).
        codigo_alternativo: Lista de códigos AEAT alternativos posibles.
        confianza: Nivel de confianza de la clasificación (0.0 a 1.0).
        justificacion: Texto explicativo con la justificación de la clasificación.
        es_deducible: Indica si el gasto es fiscalmente deducible.
        porcentaje_deducible: Porcentaje de deducibilidad como cadena (p.ej. ``'100%'``).
        oportunidades_fiscales: Lista de oportunidades de ahorro fiscal detectadas.
        alertas_cumplimiento: Lista de alertas de cumplimiento normativo.
        consultas_aplicables: Números de consultas vinculantes que respaldan la recomendación.
        confidence_score: Puntuación de confianza global del análisis.
    """
    codigo_principal: str
    codigo_alternativo: List[str]
    confianza: float
    justificacion: str
    es_deducible: bool
    porcentaje_deducible: str
    oportunidades_fiscales: List[Dict]
    alertas_cumplimiento: List[Dict]
    consultas_aplicables: List[str]
    confidence_score: float


class MinimalHashingEmbedder:
    """Embedder de respaldo sin dependencias pesadas de modelos neuronales.

    Genera vectores de dimensión fija mediante hashing MD5 de tokens. Se utiliza
    como alternativa ligera cuando ``SentenceTransformer`` no puede cargarse
    (p.ej., por errores de inicialización en dispositivo 'meta').

    El resultado no captura semántica real, pero mantiene el sistema operativo
    permitiendo búsquedas por coincidencia léxica aproximada.

    Atributos:
        dim: Dimensión del vector de salida. Por defecto 384 para mantener
             compatibilidad con ``all-MiniLM-L6-v2``.
    """

    def __init__(self, dim: int = 384):
        """Inicializa el embedder de hashing con la dimensión especificada.

        Args:
            dim: Dimensión del vector de salida. Por defecto 384 para
                mantener compatibilidad con ``all-MiniLM-L6-v2``.
        """
        self.dim = int(dim)

    def _encode_one(self, text: str) -> np.ndarray:
        """Genera un vector embedding para un único texto mediante hashing de tokens.

        Tokeniza el texto por palabras, calcula un hash MD5 por cada token y
        acumula activaciones en las posiciones correspondientes del vector.
        Finalmente aplica normalización L2.

        Args:
            text: Texto de entrada a codificar.

        Returns:
            Vector numpy de dimensión ``self.dim`` normalizado en L2.
        """
        v = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return v
        # Tokenización simple por espacios/puntuación
        tokens = re.findall(r"\w+", text.lower())
        for tok in tokens:
            # Hash estable por md5 para consistencia cross-run
            h = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dim
            v[idx] += 1.0
        # Normalización L2 para comparabilidad
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v

    def encode(self, sentences: List[str]) -> np.ndarray:
        """Codifica una lista de textos en una matriz de embeddings.

        Interfaz compatible con ``SentenceTransformer.encode()`` para que pueda
        usarse de forma intercambiable como modelo de embeddings.

        Args:
            sentences: Lista de textos a codificar. También acepta un único
                string que se envuelve automáticamente en lista.

        Returns:
            Matriz numpy de forma ``(len(sentences), self.dim)`` con los
            vectores normalizados.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
        mats = [self._encode_one(s) for s in sentences]
        return np.stack(mats, axis=0)


class FacturaDataExtractor:
    """Extractor robusto de datos de factura con manejo de campos opcionales.

    Proporciona métodos estáticos para acceder de forma segura a los distintos
    bloques de información de una factura (emisor, receptor, conceptos, contexto
    empresarial, datos fiscales y relación comercial), devolviendo valores por
    defecto cuando los campos no existen o son ``None``.

    Esta clase no mantiene estado propio; todos sus métodos son ``@staticmethod``.
    """

    @staticmethod
    def safe_get(data: Dict, *keys, default: Any = None) -> Any:
        """Extrae un valor de un diccionario anidado de forma segura.

        Navega una cadena de claves sobre diccionarios anidados y devuelve
        el valor encontrado, o ``default`` si alguna clave no existe o el
        tipo intermedio no es un diccionario.

        Args:
            data: Diccionario raíz desde donde iniciar la búsqueda.
            *keys: Secuencia de claves para navegar niveles anidados.
            default: Valor a devolver si la ruta no existe o el resultado
                es ``None``.

        Returns:
            El valor encontrado en la ruta de claves, o ``default``.
        """
        try:
            result = data
            for key in keys:
                if isinstance(result, dict) and key in result:
                    result = result[key]
                else:
                    return default
            return result if result is not None else default
        except (KeyError, TypeError, AttributeError):
            return default

    @staticmethod
    def extract_conceptos_info(factura_json: Dict) -> List[Dict]:
        """Extrae la información de los conceptos (líneas de detalle) de la factura.

        Itera sobre la lista ``conceptos`` del JSON de factura y normaliza cada
        entrada con claves estándar y valores por defecto.

        Args:
            factura_json: Diccionario con los datos completos de la factura.

        Returns:
            Lista de diccionarios, cada uno con las claves: ``descripcion``,
            ``cantidad``, ``precio_unitario``, ``importe_linea``,
            ``codigo_producto`` y ``unidad_medida``.
        """
        conceptos = FacturaDataExtractor.safe_get(factura_json, 'conceptos', default=[])
        if not isinstance(conceptos, list):
            return []

        extracted_conceptos = []
        for concepto in conceptos:
            if isinstance(concepto, dict):
                extracted_conceptos.append({
                    'descripcion': concepto.get('descripcion', ''),
                    'cantidad': concepto.get('cantidad', 1),
                    'precio_unitario': concepto.get('precio_unitario', 0),
                    'importe_linea': concepto.get('importe_linea', 0),
                    'codigo_producto': concepto.get('codigo_producto', ''),
                    'unidad_medida': concepto.get('unidad_medida', '')
                })

        return extracted_conceptos

    @staticmethod
    def extract_empresa_info(factura_json: Dict, tipo: str = 'receptor') -> Dict:
        """Extrae la información de una empresa (receptor o emisor) de la factura.

        Args:
            factura_json: Diccionario con los datos completos de la factura.
            tipo: Sección a extraer, ``'receptor'`` o ``'emisor'``.

        Returns:
            Diccionario con las claves: ``nombre``, ``cif``, ``actividad_cnae``,
            ``sector``, ``tipo_empresa`` y ``pais_residencia`` (por defecto
            ``'España'``).
        """
        empresa_data = FacturaDataExtractor.safe_get(factura_json, tipo, default={})

        return {
            'nombre': empresa_data.get('nombre', ''),
            'cif': empresa_data.get('cif', ''),
            'actividad_cnae': empresa_data.get('actividad_cnae', ''),
            'sector': empresa_data.get('sector', ''),
            'tipo_empresa': empresa_data.get('tipo_empresa', ''),
            'pais_residencia': empresa_data.get('pais_residencia', 'España')
        }

    @staticmethod
    def extract_contexto_empresarial(factura_json: Dict) -> Dict:
        """Extrae el contexto empresarial completo asociado a la factura.

        Incluye información sobre departamento, centro de coste, proyecto,
        porcentaje de afectación a la actividad, uso empresarial y
        justificación del gasto.

        Args:
            factura_json: Diccionario con los datos completos de la factura.

        Returns:
            Diccionario con las claves: ``departamento``, ``centro_coste``,
            ``proyecto``, ``porcentaje_afectacion`` (por defecto 100),
            ``uso_empresarial`` y ``justificacion_gasto``.
        """
        contexto = FacturaDataExtractor.safe_get(factura_json, 'contexto_empresarial', default={})

        return {
            'departamento': contexto.get('departamento', ''),
            'centro_coste': contexto.get('centro_coste', ''),
            'proyecto': contexto.get('proyecto', ''),
            'porcentaje_afectacion': contexto.get('porcentaje_afectacion', 100),
            'uso_empresarial': contexto.get('uso_empresarial', ''),
            'justificacion_gasto': contexto.get('justificacion_gasto', '')
        }

    @staticmethod
    def extract_fiscal_info(factura_json: Dict) -> Dict:
        """Extrae la información fiscal específica de la factura.

        Incluye régimen de IVA, retención aplicada, tipo de retención,
        si es operación intracomunitaria y si aplica alguna exención.

        Args:
            factura_json: Diccionario con los datos completos de la factura.

        Returns:
            Diccionario con las claves: ``regimen_iva`` (por defecto
            ``'General'``), ``retencion_aplicada``, ``tipo_retencion``,
            ``operacion_intracomunitaria`` y ``exencion_aplicada``.
        """
        fiscal = FacturaDataExtractor.safe_get(factura_json, 'fiscal', default={})

        return {
            'regimen_iva': fiscal.get('regimen_iva', 'General'),
            'retencion_aplicada': fiscal.get('retencion_aplicada', False),
            'tipo_retencion': fiscal.get('tipo_retencion', 0),
            'operacion_intracomunitaria': fiscal.get('operacion_intracomunitaria', False),
            'exencion_aplicada': fiscal.get('exencion_aplicada', False)
        }

    @staticmethod
    def extract_relacion_comercial(factura_json: Dict) -> Dict:
        """Extrae la información sobre la relación comercial entre las partes.

        Determina si se trata de un tercero independiente, empresa vinculada,
        operación vinculada o proveedor habitual.

        Args:
            factura_json: Diccionario con los datos completos de la factura.

        Returns:
            Diccionario con las claves: ``tipo_relacion`` (por defecto
            ``'Tercero independiente'``), ``empresa_vinculada``,
            ``operacion_vinculada`` y ``proveedor_habitual``.
        """
        relacion = FacturaDataExtractor.safe_get(factura_json, 'relacion_comercial', default={})

        return {
            'tipo_relacion': relacion.get('tipo_relacion', 'Tercero independiente'),
            'empresa_vinculada': relacion.get('empresa_vinculada', False),
            'operacion_vinculada': relacion.get('operacion_vinculada', False),
            'proveedor_habitual': relacion.get('proveedor_habitual', False)
        }


class ConsultasTributariasProcessor:
    """Procesa consultas tributarias vinculantes oficiales de la AEAT usando VLLM.

    Responsable de:
        - Cargar un modelo de embeddings semánticos (con fallback a hashing).
        - Analizar el texto de cada consulta tributaria mediante el modelo de
          lenguaje VLLM para extraer patrones fiscales estructurados.
        - Generar embeddings de cada consulta para posterior búsqueda por
          similitud.

    Atributos:
        vllm: Adaptador LangChain para el modelo de lenguaje VLLM.
        embedding_model: Modelo de embeddings (``SentenceTransformer`` o
            ``MinimalHashingEmbedder`` como respaldo).
        codigo_patterns: Diccionario que mapea cada código AEAT a una lista
            de palabras clave asociadas, utilizado en el análisis de fallback.
    """

    def __init__(self):
        """Inicializa el procesador de consultas tributarias.

        Configura el adaptador VLLM con temperatura baja (0.2) para respuestas
        deterministas, carga el modelo de embeddings y define el diccionario
        de patrones de palabras clave por código AEAT para el análisis de
        fallback.
        """
        self.vllm = LangChainVLLMAdapter(temperature=0.2, max_tokens=6000)

        # Carga robusta de SentenceTransformer con fallback para entornos que usan 'meta' device
        self.embedding_model = self._load_embedding_model()

        # Patrones específicos por códigos AEAT - Lista completa
        self.codigo_patterns = {
            # Existencias y compras
            'G01': ['compra', 'existencias', 'mercancías', 'materias primas', 'stock', 'inventario', 'productos',
                    'adquisición'],
            'G02': ['variación existencias', 'ajuste inventario', 'disminución existencias', 'diferencia inventario'],

            # Consumos y suministros
            'G03': ['combustible', 'material oficina', 'consumos', 'suministros menores', 'papelería', 'envases',
                    'embalajes'],

            # Personal y nóminas
            'G04': ['sueldos', 'salarios', 'nómina', 'retribución', 'paga', 'sueldo'],
            'G05': ['seguridad social empresa', 'cotizaciones empresa', 'ss empresa', 'cotización trabajadores'],
            'G06': ['seguridad social autónomo', 'reta', 'mutualidad', 'cotización autónomo', 'ss autónomo'],
            'G45': ['seguridad social titular', 'cotización titular', 'ss titular actividad'],
            'G46': ['mutualidad alternativa', 'aportación mutualidad', 'mutua alternativa'],
            'G07': ['indemnización', 'despido', 'finiquito', 'compensación'],
            'G08': ['dietas', 'viajes personal', 'desplazamientos', 'kilometraje', 'gastos viaje'],
            'G09': ['previsión social', 'plan pensiones empleados', 'aportaciones empleados'],
            'G10': ['otros gastos personal', 'formación', 'uniformes', 'equipamiento personal'],
            'G11': ['manutención contribuyente', 'dietas autónomo', 'gastos manutención'],

            # Arrendamientos y servicios
            'G12': ['alquiler', 'arrendamiento', 'canon', 'software', 'licencia', 'renting', 'leasing'],
            'G13': ['reparación', 'conservación', 'mantenimiento', 'reparaciones', 'arreglo'],
            'G14': ['electricidad', 'luz', 'suministro eléctrico', 'energía eléctrica'],
            'G15': ['tributos', 'tasas', 'impuestos', 'licencias municipales', 'icio'],
            'G16': ['gastos financieros', 'intereses', 'comisiones bancarias', 'gastos banco'],
            'G17': ['telefonía', 'internet', 'comunicaciones', 'móvil', 'teléfono', 'datos'],
            'G18': ['otros suministros', 'agua', 'gas', 'suministros varios'],

            # Servicios profesionales
            'G19': ['consultoría', 'servicios profesionales', 'asesoría', 'auditoría', 'desarrollo', 'abogado',
                    'gestor'],
            'G20': ['publicidad', 'propaganda', 'marketing', 'relaciones públicas', 'promoción'],
            'G21': ['transporte', 'portes', 'envío', 'courier', 'mensajería', 'logística'],
            'G22': ['seguros', 'póliza', 'prima seguro', 'seguro responsabilidad'],
            'G25': ['iva no deducible', 'iva soportado', 'recargo equivalencia'],

            # Servicios externos
            'G42': ['limpieza', 'seguridad', 'vigilancia', 'otros servicios', 'servicios terceros'],

            # Amortizaciones y otros
            'G30': ['amortización', 'depreciación', 'amortización inmovilizado'],
            'G31': ['deterioro', 'provisión', 'pérdida valor'],
            'G32': ['otros gastos gestión', 'gastos diversos', 'varios'],
            'G33': ['gastos excepcionales', 'extraordinarios', 'no recurrentes'],

            # Códigos adicionales comunes
            'G23': ['servicios bancarios', 'comisiones', 'gastos cuenta'],
            'G24': ['investigación desarrollo', 'i+d', 'innovación'],
            'G26': ['dotaciones amortización', 'amortización ejercicio'],
            'G27': ['pérdidas créditos', 'insolvencias', 'morosos'],
            'G28': ['variación provisiones', 'dotación provisión'],
            'G29': ['otros gastos explotación', 'gastos explotación varios']
        }

        logger.info("ConsultasTributariasProcessor iniciado")

    @staticmethod
    def _load_embedding_model():
        """Carga un modelo de embeddings de forma robusta evitando errores con 'meta' device.

        Intenta forzar CPU y deshabilitar low_cpu_mem_usage para evitar inicialización en 'meta'.
        Si falla, usa un embedder hash mínimo como fallback para mantener el sistema operativo.
        """
        model_name = 'all-MiniLM-L6-v2'
        try:
            # Evita inicialización en 'meta' pasando low_cpu_mem_usage=False y usando CPU
            return SentenceTransformer(model_name, device='cpu', model_kwargs={'low_cpu_mem_usage': False})
        except NotImplementedError as e:
            logger.warning(f"Fallo al cargar {model_name} (meta tensor): {e}. Reintentando sin 'device' explícito...")
            try:
                return SentenceTransformer(model_name, model_kwargs={'low_cpu_mem_usage': False})
            except Exception as e2:
                logger.error(f"Fallo al cargar {model_name} nuevamente: {e2}. Usando fallback hash embedder.")
                return MinimalHashingEmbedder()
        except Exception as e:
            logger.error(f"No se pudo cargar {model_name}: {e}. Usando fallback hash embedder.")
            return MinimalHashingEmbedder()

    def process_consultas_batch(self, consultas_json: List[Dict]) -> List[Dict]:
        """Procesa un lote de consultas tributarias extrayendo patrones fiscales.

        Para cada consulta del lote, invoca el análisis fiscal por lotes con VLLM,
        genera un texto optimizado para búsqueda semántica, calcula su embedding
        y devuelve la consulta enriquecida con estos campos adicionales.

        Args:
            consultas_json: Lista de diccionarios, cada uno representando una
                consulta tributaria con campos como ``numero_consulta``,
                ``descripcion_hechos``, ``contestacion_completa``, etc.

        Returns:
            Lista de diccionarios enriquecidos con las claves adicionales:
            ``fiscal_analysis``, ``embedding`` (bytes), ``searchable_text``
            y ``processed_timestamp``.
        """
        logger.info(f"Procesando {len(consultas_json)} consultas tributarias...")

        processed_consultas = []
        # Extraer patrones fiscales usando VLLM
        batch_fiscal_analysis = self.extract_fiscal_patterns_in_batch(consultas_json)

        for i, (consulta, fiscal_analysis) in enumerate(zip(consultas_json, batch_fiscal_analysis)):
            try:
                logger.info(
                    f"Procesando consulta {i + 1}/{len(consultas_json)}: {consulta.get('numero_consulta', 'N/A')}")

                # Crear texto para búsqueda semántica
                searchable_text = self.create_searchable_text(consulta)

                # Generar embedding para similaridad
                embedding = self.embedding_model.encode([searchable_text])[0]

                processed_consulta = {
                    **consulta,
                    'fiscal_analysis': fiscal_analysis,
                    'embedding': embedding.tobytes(),
                    'searchable_text': searchable_text,
                    'processed_timestamp': datetime.now().isoformat()
                }

                processed_consultas.append(processed_consulta)

            except Exception as e:
                logger.error(f"Error procesando consulta {consulta.get('numero_consulta', 'N/A')}: {e}")
                continue

        logger.info(f"Procesadas exitosamente {len(processed_consultas)} consultas")
        return processed_consultas

    def extract_fiscal_patterns(self, consulta: Dict) -> Dict:
        """Extrae patrones fiscales de una consulta individual usando VLLM.

        Construye un prompt con los hechos y la resolución de la consulta,
        invoca al modelo de lenguaje y parsea la respuesta JSON con los
        patrones fiscales detectados.

        Si falla el parsing JSON o la llamada al modelo, recurre al método
        de análisis de fallback basado en coincidencia de palabras clave.

        Args:
            consulta: Diccionario con los datos de la consulta tributaria.

        Returns:
            Diccionario con claves: ``codigos_aeat_aplicables``,
            ``criterios_deducibilidad``, ``incentivos_detectados``,
            ``cnaes_relevantes``, ``condiciones_especiales``,
            ``alertas_importantes`` y ``keywords_fiscales``.
        """

        # Limitar longitud para evitar truncamiento
        hechos = consulta.get('descripcion_hechos', '')[:1200]
        resolucion = consulta.get('contestacion_completa', '')[:1500]

        prompt = f"""
Analiza esta consulta tributaria oficial de la AEAT y extrae los patrones fiscales clave.

CONSULTA: {consulta.get('numero_consulta', 'N/A')}
ÓRGANO: {consulta.get('organo', 'N/A')}
NORMATIVA: {consulta.get('normativa', 'N/A')}

HECHOS RELEVANTES:
{hechos}

RESOLUCIÓN OFICIAL:
{resolucion}

Extrae los patrones fiscales más importantes. Responde ÚNICAMENTE en formato JSON válido:

{{
    "codigos_aeat_aplicables": ["G19", "G03"],
    "criterios_deducibilidad": ["vinculación con actividad", "documentación correcta"],
    "incentivos_detectados": ["I+D+i 25%", "régimen especial", "exención participaciones"],
    "cnaes_relevantes": ["6201", "4711", "7211"],
    "condiciones_especiales": ["participación superior 5%", "tenencia 1 año"],
    "alertas_importantes": ["verificar vinculación", "documentación adicional"],
    "keywords_fiscales": ["aportación no dineraria", "motivos económicos válidos", "neutralidad fiscal"]
}}
"""

        try:
            response = self.vllm.generate(prompt)
            if '</think>' in response:
                response = response.split('</think')[-1]
            cleaned_response = self.clean_json_response(response)
            return json.loads(cleaned_response)

        except json.JSONDecodeError as e:
            logger.warning(f"Error JSON en consulta {consulta.get('numero_consulta', 'N/A')}: {e}")
            return self.fallback_analysis(consulta)
        except Exception as e:
            logger.error(f"Error VLLM en consulta {consulta.get('numero_consulta', 'N/A')}: {e}")
            return self.fallback_analysis(consulta)

    def extract_fiscal_patterns_in_batch(self, consultas: list) -> list:
        """Extrae patrones fiscales de un lote de consultas usando VLLM en modo batch.

        Construye un prompt para cada consulta y los envía simultáneamente
        al modelo de lenguaje mediante ``batch_generate_async_wrapper``.
        Esto es significativamente más eficiente que procesar una a una.

        Si falla el procesamiento del lote, recurre a ``fallback_analysis``
        para cada consulta individualmente.

        Args:
            consultas: Lista de diccionarios con los datos de las consultas
                tributarias.

        Returns:
            Lista de diccionarios con los patrones fiscales extraídos,
            en el mismo orden que las consultas de entrada.
        """

        batch_prompt = []
        for consulta in consultas:
            # Limitar longitud para evitar truncamiento
            hechos = consulta.get('descripcion_hechos', '')[:2000]
            resolucion = consulta.get('contestacion_completa', '')[:2000]

            prompt = f"""
Analiza esta consulta tributaria oficial de la AEAT y extrae los patrones fiscales clave.

CONSULTA: {consulta.get('numero_consulta', 'N/A')}
ÓRGANO: {consulta.get('organo', 'N/A')}
NORMATIVA: {consulta.get('normativa', 'N/A')}

HECHOS RELEVANTES:
{hechos}

RESOLUCIÓN OFICIAL:
{resolucion}

Extrae los patrones fiscales más importantes. Responde ÚNICAMENTE en formato JSON válido:

{{
    "codigos_aeat_aplicables": ["G19", "G03"],
    "criterios_deducibilidad": ["vinculación con actividad", "documentación correcta"],
    "incentivos_detectados": ["I+D+i 25%", "régimen especial", "exención participaciones"],
    "cnaes_relevantes": ["6201", "4711", "7211"],
    "condiciones_especiales": ["participación superior 5%", "tenencia 1 año"],
    "alertas_importantes": ["verificar vinculación", "documentación adicional"],
    "keywords_fiscales": ["aportación no dineraria", "motivos económicos válidos", "neutralidad fiscal"]
}}
"""
            batch_prompt += [prompt]

        try:
            batch_response: list = self.vllm.batch_generate_async_wrapper(prompts=batch_prompt)
            batch_cleaned_response = [self.clean_json_response(response) for response in batch_response]

            return [json.loads(cleaned_response) for cleaned_response in batch_cleaned_response]

        except json.JSONDecodeError as e:
            logger.warning(f"Error JSON en batch consulta: {e}")
            return [self.fallback_analysis(consulta) for consulta in consultas]
        except Exception as e:
            logger.error(f"Error VLLM en batch consulta: {e}")
            return [self.fallback_analysis(consulta) for consulta in consultas]


    def clean_json_response(self, response: str) -> str:
        """Limpia la respuesta cruda del VLLM para extraer un JSON válido.

        Localiza las llaves de apertura y cierre del objeto JSON dentro del
        texto de respuesta y elimina caracteres de control que podrían
        invalidar el parsing.

        Args:
            response: Texto de respuesta completo del modelo de lenguaje.

        Returns:
            Cadena con el JSON limpio listo para ``json.loads()``.

        Raises:
            ValueError: Si no se encuentra un objeto JSON en la respuesta.
        """
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx != -1 and end_idx != 0:
            json_str = response[start_idx:end_idx]
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
            json_str = re.sub(r'\\n', ' ', json_str)
            json_str = re.sub(r'\\t', ' ', json_str)
            return json_str

        raise ValueError("No se encontró JSON válido en respuesta")

    def fallback_analysis(self, consulta: Dict) -> Dict:
        """Análisis de respaldo usando coincidencia simple de palabras clave.

        Se invoca cuando el modelo de lenguaje falla o devuelve una respuesta
        no parseable. Busca coincidencias entre el texto de la consulta y los
        patrones de ``self.codigo_patterns``, además de extraer códigos CNAE
        mediante expresión regular.

        Args:
            consulta: Diccionario con los datos de la consulta tributaria.

        Returns:
            Diccionario con la misma estructura que ``extract_fiscal_patterns``,
            pero con datos parciales y la marca ``'fallback_analysis'`` en
            ``keywords_fiscales``.
        """
        logger.info(f"Usando análisis de fallback para {consulta.get('numero_consulta', 'N/A')}")

        text_content = (
                consulta.get('descripcion_hechos', '') + ' ' +
                consulta.get('contestacion_completa', '')
        ).lower()

        # Detectar códigos probables
        codigos_detectados = []
        for codigo, keywords in self.codigo_patterns.items():
            if any(keyword in text_content for keyword in keywords):
                codigos_detectados.append(codigo)

        # Extraer CNAEs mencionados
        cnaes = re.findall(r'\b\d{4}\b', text_content)

        return {
            "codigos_aeat_aplicables": codigos_detectados[:3],
            "criterios_deducibilidad": ["vinculación actividad"],
            "incentivos_detectados": [],
            "cnaes_relevantes": list(set(cnaes))[:5],
            "condiciones_especiales": [],
            "alertas_importantes": ["análisis automático"],
            "keywords_fiscales": ["fallback_analysis"]
        }

    def create_searchable_text(self, consulta: Dict) -> str:
        """Crea un texto concatenado y optimizado para búsqueda semántica.

        Combina los campos más relevantes de la consulta (número, órgano,
        normativa, hechos y conclusiones clave) en un único texto que será
        codificado como embedding para comparación por similitud coseno.

        Args:
            consulta: Diccionario con los datos de la consulta tributaria.

        Returns:
            Cadena de texto con la información relevante concatenada.
        """
        parts = []

        if consulta.get('numero_consulta'):
            parts.append(f"Consulta {consulta['numero_consulta']}")

        if consulta.get('organo'):
            parts.append(consulta['organo'])

        if consulta.get('normativa'):
            parts.append(f"Normativa: {consulta['normativa']}")

        hechos = consulta.get('descripcion_hechos', '')
        if hechos:
            parts.append(f"Hechos: {hechos[:500]}")

        resolucion = consulta.get('contestacion_completa', '')
        if resolucion:
            conclusiones = self.extract_key_conclusions(resolucion)
            parts.append(f"Resolución: {conclusiones}")

        return ' '.join(parts)

    def extract_key_conclusions(self, resolucion: str) -> str:
        """Extrae las conclusiones clave de un texto de resolución oficial.

        Busca oraciones que contengan palabras clave indicativas de conclusión
        jurídica (p.ej. ``'por tanto'``, ``'será deducible'``) y devuelve
        las primeras tres encontradas.

        Args:
            resolucion: Texto completo de la resolución/contestación oficial.

        Returns:
            Cadena con hasta 3 oraciones conclusivas, truncada a 800 caracteres.
        """
        conclusiones_keywords = [
            'por tanto', 'en consecuencia', 'se aplicará', 'resultará de aplicación',
            'estará exenta', 'será deducible', 'no se integrará'
        ]

        sentences = resolucion.split('.')
        key_sentences = []

        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(keyword in sentence_lower for keyword in conclusiones_keywords):
                key_sentences.append(sentence.strip())
                if len(key_sentences) >= 3:
                    break

        return '. '.join(key_sentences)[:800]


class ConsultasRecommendationEngine:
    """Motor de recomendaciones fiscales basado en consultas tributarias vinculantes.

    Dado un conjunto de consultas tributarias ya procesadas (con embeddings y
    análisis fiscal), este motor:
        1. Busca las consultas más relevantes para una factura dada mediante
           similitud coseno + boosting contextual.
        2. Construye un prompt enriquecido con los precedentes oficiales.
        3. Invoca al VLLM para generar la recomendación de clasificación.
        4. Enriquece el resultado con metadatos, ahorros estimados y
           factores de riesgo.

    Atributos:
        vllm: Adaptador LangChain para el modelo de lenguaje VLLM.
        consultas_db: Lista de consultas tributarias procesadas con embeddings.
        embedding_model: Modelo de embeddings para codificar textos de factura.
        extractor: Instancia de ``FacturaDataExtractor`` para acceso seguro
            a los campos de factura.
    """

    def __init__(self, processed_consultas: List[Dict]):
        """Inicializa el motor de recomendaciones con las consultas procesadas.

        Configura el adaptador VLLM con temperatura moderada (0.3), carga el
        modelo de embeddings y almacena la base de datos de consultas
        tributarias ya procesadas.

        Args:
            processed_consultas: Lista de consultas tributarias procesadas,
                cada una conteniendo su embedding y análisis fiscal.
        """
        self.vllm = LangChainVLLMAdapter(temperature=0.3, max_tokens=6000)

        self.consultas_db = processed_consultas
        # Reusar el mismo cargador robusto que en ConsultasTributariasProcessor
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu',
                                                       model_kwargs={'low_cpu_mem_usage': False})
        except Exception:
            self.embedding_model = MinimalHashingEmbedder()
        self.extractor = FacturaDataExtractor()

        logger.info(f"Motor de recomendaciones iniciado con {len(processed_consultas)} consultas")

    def get_best_recommendation(self, factura_json: Dict) -> Dict:
        """Genera la mejor recomendación fiscal para una factura dada.

        Orquesta el flujo completo: búsqueda de consultas relevantes,
        generación de recomendación contextual con VLLM y enriquecimiento
        con metadatos y factores de riesgo. Si ocurre un error en cualquier
        paso, devuelve una recomendación de fallback.

        Args:
            factura_json: Diccionario con los datos completos de la factura.

        Returns:
            Diccionario con la recomendación fiscal completa incluyendo
            clasificación, deducibilidad, oportunidades, alertas y metadatos.
        """
        logger.info("Generando recomendación fiscal...")

        try:
            # Encontrar consultas más relevantes
            relevant_consultas = self.find_most_relevant_consultas(factura_json, top_k=10)
            logger.info(f"Encontradas {len(relevant_consultas)} consultas relevantes")

            # Generar recomendación contextual
            recommendation = self.generate_contextual_recommendation(
                factura_json, relevant_consultas
            )

            # Enriquecer con análisis adicional
            enhanced_recommendation = self.enrich_recommendation(
                recommendation, factura_json, relevant_consultas
            )

            return enhanced_recommendation

        except Exception as e:
            logger.error(f"Error generando recomendación: {e}")
            return self.fallback_recommendation()

    def find_most_relevant_consultas(self, factura_json: Dict, top_k: int = 5) -> List[Dict]:
        """Encuentra las consultas tributarias más relevantes para una factura.

        Calcula la similitud coseno entre el embedding de la factura y el de
        cada consulta almacenada, aplica un factor de boost basado en
        coincidencias contextuales (CNAE, keywords, departamento, tipo de
        empresa, etc.) y devuelve las ``top_k`` consultas con mayor
        puntuación final.

        Args:
            factura_json: Diccionario con los datos completos de la factura.
            top_k: Número máximo de consultas relevantes a devolver.

        Returns:
            Lista de diccionarios de consultas ordenados por relevancia
            descendente, limitada a ``top_k`` elementos.
        """

        factura_text = self.create_factura_search_text(factura_json)
        factura_embedding = self.embedding_model.encode([factura_text])[0]

        similarities = []

        for consulta in self.consultas_db:
            try:
                consulta_embedding = np.frombuffer(consulta['embedding'], dtype=np.float32)
                similarity = np.dot(factura_embedding, consulta_embedding) / (
                        np.linalg.norm(factura_embedding) * np.linalg.norm(consulta_embedding)
                )

                boost = self.calculate_relevance_boost(factura_json, consulta)
                final_score = similarity + boost

                similarities.append({
                    'consulta': consulta,
                    'similarity': float(similarity),
                    'boost': float(boost),
                    'final_score': float(final_score)
                })

            except Exception as e:
                logger.warning(f"Error procesando consulta {consulta.get('numero_consulta', 'N/A')}: {e}")
                continue

        similarities.sort(key=lambda x: x['final_score'], reverse=True)

        #logger.info(
        #    f"Top 3 consultas: {[s['consulta']['numero_consulta']} score: {s['final_score']:.3f}" for s in similarities[:3]]}")

        return [s['consulta'] for s in similarities[:top_k]]

    def create_factura_search_text(self, factura_json: Dict) -> str:
        """Crea un texto de búsqueda optimizado a partir de la factura.

        Concatena los campos más relevantes de la factura (CNAEs, tipo de
        empresa, sector, descripciones de conceptos, contexto empresarial,
        información fiscal y relación comercial) en un único texto que será
        codificado como embedding para búsqueda por similitud.

        Args:
            factura_json: Diccionario con los datos completos de la factura.

        Returns:
            Cadena de texto con la información relevante concatenada.
        """
        parts = []

        # Información del emisor
        emisor = self.extractor.extract_empresa_info(factura_json, 'emisor')
        if emisor['actividad_cnae']:
            parts.append(f"CNAE emisor {emisor['actividad_cnae']}")

        # Información del receptor
        receptor = self.extractor.extract_empresa_info(factura_json, 'receptor')
        if receptor['actividad_cnae']:
            parts.append(f"CNAE receptor {receptor['actividad_cnae']}")
        if receptor['tipo_empresa']:
            parts.append(f"Tipo empresa {receptor['tipo_empresa']}")
        if receptor['sector']:
            parts.append(f"Sector {receptor['sector']}")

        # Conceptos detallados
        conceptos = self.extractor.extract_conceptos_info(factura_json)
        for concepto in conceptos:
            if concepto['descripcion']:
                parts.append(f"Concepto: {concepto['descripcion']}")
            if concepto['codigo_producto']:
                parts.append(f"Código producto: {concepto['codigo_producto']}")

        # Contexto empresarial enriquecido
        contexto = self.extractor.extract_contexto_empresarial(factura_json)
        if contexto['departamento']:
            parts.append(f"Departamento {contexto['departamento']}")
        if contexto['proyecto']:
            parts.append(f"Proyecto {contexto['proyecto']}")
        if contexto['uso_empresarial']:
            parts.append(f"Uso {contexto['uso_empresarial']}")
        if contexto['justificacion_gasto']:
            parts.append(f"Justificación: {contexto['justificacion_gasto']}")

        # Información fiscal
        fiscal = self.extractor.extract_fiscal_info(factura_json)
        if fiscal['retencion_aplicada']:
            parts.append(f"Con retención {fiscal['tipo_retencion']}%")

        # Relación comercial
        relacion = self.extractor.extract_relacion_comercial(factura_json)
        if relacion['operacion_vinculada']:
            parts.append("Operación vinculada")

        return ' '.join(parts)

    def calculate_relevance_boost(self, factura_json: Dict, consulta: Dict) -> float:
        """Calcula un factor de boost de relevancia entre factura y consulta.

        Evalúa múltiples señales contextuales para incrementar la puntuación
        de similitud base:
            - Coincidencia de CNAE (hasta +0.25).
            - Keywords fiscales en conceptos (hasta +0.18).
            - Tipo de empresa coincidente (+0.10).
            - Departamento especializado (+0.12).
            - Proyecto específico mencionado (+0.08).
            - Operaciones vinculadas (+0.15).
            - Uso empresarial exclusivo (+0.05).

        El boost total se limita a un máximo de 0.6 (60%).

        Args:
            factura_json: Diccionario con los datos completos de la factura.
            consulta: Diccionario de una consulta tributaria procesada.

        Returns:
            Valor float del boost a sumar a la similitud coseno base.
        """
        boost = 0.0

        fiscal_analysis = consulta.get('fiscal_analysis', {})
        receptor = self.extractor.extract_empresa_info(factura_json, 'receptor')
        contexto = self.extractor.extract_contexto_empresarial(factura_json)
        relacion = self.extractor.extract_relacion_comercial(factura_json)

        # Boost por CNAE coincidente (mejorado)
        if receptor['actividad_cnae']:
            cnae_clean = receptor['actividad_cnae'].split(' - ')[0] if ' - ' in receptor['actividad_cnae'] else \
            receptor['actividad_cnae']
            cnae_2d = cnae_clean[:2] if len(cnae_clean) >= 2 else cnae_clean

            cnaes_relevantes = fiscal_analysis.get('cnaes_relevantes', [])
            for cnae in cnaes_relevantes:
                if cnae.startswith(cnae_2d):
                    boost += 0.25  # Aumentado por mayor precisión
                    break

        # Boost por keywords en conceptos (mejorado)
        conceptos = self.extractor.extract_conceptos_info(factura_json)
        conceptos_text = ' '.join([c['descripcion'].lower() for c in conceptos if c['descripcion']])

        keywords_fiscales = fiscal_analysis.get('keywords_fiscales', [])
        matched_keywords = sum(1 for keyword in keywords_fiscales if keyword.lower() in conceptos_text)

        if matched_keywords > 0:
            boost += min(matched_keywords * 0.06, 0.18)  # Max 0.18

        # Boost por tipo de empresa específico
        tipo_empresa = receptor['tipo_empresa'].lower() if receptor['tipo_empresa'] else ''
        searchable_text = consulta.get('searchable_text', '').lower()

        if 'sl' in tipo_empresa and 'sociedad' in searchable_text:
            boost += 0.1

        # Boost por departamento especializado (mejorado)
        departamento = contexto['departamento']
        if departamento:
            dept_keywords = {
                'I+D': ['investigación', 'desarrollo', 'innovación', 'i+d+i'],
                'IT': ['informática', 'tecnología', 'software', 'sistemas'],
                'Marketing': ['publicidad', 'promoción', 'marketing', 'comercial'],
                'Producción': ['fabricación', 'producción', 'manufacturera'],
                'Financiero': ['financiero', 'contabilidad', 'fiscal']
            }

            if departamento in dept_keywords:
                for keyword in dept_keywords[departamento]:
                    if keyword in searchable_text:
                        boost += 0.12
                        break

        # Boost por proyecto específico
        if contexto['proyecto'] and len(contexto['proyecto']) > 3:
            if contexto['proyecto'].lower() in searchable_text:
                boost += 0.08

        # Boost por operaciones vinculadas
        if relacion['operacion_vinculada'] and 'vinculada' in searchable_text:
            boost += 0.15

        # Boost por uso empresarial exclusivo
        if contexto['uso_empresarial'] == 'Exclusivo':
            boost += 0.05

        return min(boost, 0.6)  # Máximo boost del 60%

    def generate_contextual_recommendation(self, factura_json: Dict,
                                           relevant_consultas: List[Dict]) -> Dict:
        """Genera una recomendación fiscal usando VLLM con precedentes oficiales.

        Construye un prompt detallado que incluye el contexto enriquecido de
        la factura y los precedentes de las consultas vinculantes más
        relevantes. El modelo de lenguaje devuelve una clasificación
        estructurada en JSON.

        Args:
            factura_json: Diccionario con los datos completos de la factura.
            relevant_consultas: Lista de consultas tributarias relevantes
                previamente seleccionadas por similitud.

        Returns:
            Diccionario con la recomendación fiscal en formato estructurado
            (clasificación, deducibilidad, oportunidades, alertas, etc.).
            En caso de error, devuelve una recomendación de fallback.
        """

        precedentes_context = self.build_precedentes_context(relevant_consultas)
        factura_context = self.build_enhanced_factura_context(factura_json)

        prompt = f"""
Eres un experto fiscal español especializado en códigos AEAT. Analiza esta factura empresarial usando los precedentes oficiales de consultas vinculantes de la AEAT.

FACTURA A ANALIZAR:
{factura_context}

PRECEDENTES OFICIALES RELEVANTES (Consultas Vinculantes AEAT):
{precedentes_context}

INSTRUCCIONES:
1. Clasifica el gasto según códigos AEAT G01-G46
2. Evalúa deducibilidad considerando contexto empresarial específico
3. Identifica oportunidades fiscales con estimación económica precisa
4. Considera relaciones comerciales y aspectos fiscales especiales
5. Cita consultas vinculantes aplicables como justificación legal
6. Alerta sobre riesgos específicos según documentación y contexto

IMPORTANTE - CÁLCULO DE AHORROS FISCALES:
- Si identificas incentivo I+D+i en las consultas vinculantes:
  * Extrae el porcentaje de deducción mencionado en la normativa
  * Calcula: ahorro_estimado_euros = base_imponible × (porcentaje_deducción/100)
  * USA la base_imponible REAL de la factura analizada
  * NO uses valores de ejemplo, CALCULA el valor exacto
- Si NO aplican incentivos, establece ahorro_estimado_euros en 0
- Para otros incentivos fiscales, aplica la misma lógica de cálculo dinámico

Responde ÚNICAMENTE en formato JSON válido:

{{
    "clasificacion": {{
        "codigo_principal": "GXX",
        "codigo_alternativo": ["GYY", "GZZ"],
        "confianza": 0.XX,
        "justificacion": "Según consulta VXXXX-XX, servicios de consultoría informática se clasifican como G19 cuando..."
    }},
    "deducibilidad": {{
        "es_deducible": true,
        "porcentaje_deducible": "100%",
        "condiciones_cumplimiento": ["vinculación con actividad según VXXXX-XX", "uso empresarial exclusivo"],
        "documentacion_requerida": ["factura detallada", "contrato servicios", "informe horas"],
        "verificaciones_adicionales": ["revisar código CNAE coincidente"]
    }},
    "oportunidades_fiscales": [
        {{
            "tipo": "incentivo_idi",
            "descripcion": "Desarrollo ERP puede calificar para incentivos I+D+i",
            "ahorro_estimado_euros": 0,
            "porcentaje_deduccion": "25%",
            "requisitos_especificos": ["justificar carácter innovador", "documentar I+D"],
            "precedente_oficial": "VXXXX-XX",
            "normativa_aplicable": "Art. 35 Ley 27/2014",
            "aplicabilidad": "alta"
        }}
    ],
    "alertas_cumplimiento": [
        {{
            "tipo": "documentacion",
            "descripcion": "Verificar documentación soporte según VXXXX-XX",
            "nivel_riesgo": "medio",
            "accion_requerida": "conservar contrato e informe horas detallado",
            "plazo_accion": "inmediato"
        }}
    ],
    "analisis_relacion_comercial": {{
        "riesgo_vinculacion": "bajo",
        "recomendaciones": ["operación a precio de mercado"]
    }},
    "consultas_vinculantes_aplicables": ["VXXXX-XX", "VYYYY-YY"],
    "confidence_score": 0.XX
}}
"""

        try:
            response = self.vllm.generate(prompt)
            if '</think>' in response:
                response = response.split('</think')[-1]
            cleaned_response = self.clean_json_response(response)
            return json.loads(cleaned_response)

        except Exception as e:
            logger.error(f"Error generando recomendación con VLLM: {e}")
            return self.fallback_recommendation()

    def build_enhanced_factura_context(self, factura_json: Dict) -> str:
        """Construye un texto de contexto enriquecido de la factura para el prompt.

        Formatea de manera legible toda la información disponible de la factura
        (identificación, emisor, receptor, conceptos detallados, importes,
        contexto empresarial, información fiscal, relación comercial y fechas)
        para incluirla en el prompt del modelo de lenguaje.

        Args:
            factura_json: Diccionario con los datos completos de la factura.

        Returns:
            Cadena multilínea con el contexto formateado de la factura.
        """
        lines = []

        # Identificación
        identificacion = self.extractor.safe_get(factura_json, 'identificacion', default={})
        lines.append(
            f"FACTURA: {identificacion.get('numero_factura', 'N/A')} - Serie {identificacion.get('serie', 'N/A')}")

        # Emisor
        emisor = self.extractor.extract_empresa_info(factura_json, 'emisor')
        lines.append(f"\nEMISOR: {emisor['nombre']}")
        lines.append(f"- CIF: {emisor['cif']}")
        lines.append(f"- CNAE: {emisor['actividad_cnae']}")
        lines.append(f"- País: {emisor['pais_residencia']}")

        # Receptor
        receptor = self.extractor.extract_empresa_info(factura_json, 'receptor')
        lines.append(f"\nRECEPTOR: {receptor['nombre']}")
        lines.append(f"- CIF: {receptor['cif']}")
        lines.append(f"- CNAE: {receptor['actividad_cnae']}")
        lines.append(f"- Sector: {receptor['sector']}")
        lines.append(f"- Tipo: {receptor['tipo_empresa']}")

        # Conceptos detallados
        lines.append("\nCONCEPTOS:")
        conceptos = self.extractor.extract_conceptos_info(factura_json)
        for i, concepto in enumerate(conceptos, 1):
            lines.append(f"{i}. {concepto['descripcion']}")
            lines.append(f"   - Cantidad: {concepto['cantidad']} {concepto['unidad_medida']}")
            lines.append(f"   - Precio unitario: €{concepto['precio_unitario']}")
            lines.append(f"   - Importe: €{concepto['importe_linea']}")
            if concepto['codigo_producto']:
                lines.append(f"   - Código producto: {concepto['codigo_producto']}")

        # Importes
        importes = self.extractor.safe_get(factura_json, 'importes', default={})
        lines.append(f"\nIMPORTES:")
        lines.append(f"- Base imponible: €{importes.get('base_imponible', '0')}")
        lines.append(f"- IVA 21%: €{importes.get('iva_21', '0')}")
        lines.append(f"- IRPF: €{importes.get('irpf', '0')}")
        lines.append(f"- TOTAL: €{importes.get('total_factura', '0')}")

        # Contexto empresarial
        contexto = self.extractor.extract_contexto_empresarial(factura_json)
        lines.append(f"\nCONTEXTO EMPRESARIAL:")
        lines.append(f"- Departamento: {contexto['departamento']}")
        lines.append(f"- Centro coste: {contexto['centro_coste']}")
        lines.append(f"- Proyecto: {contexto['proyecto']}")
        lines.append(f"- Afectación: {contexto['porcentaje_afectacion']}%")
        lines.append(f"- Uso: {contexto['uso_empresarial']}")
        if contexto['justificacion_gasto']:
            lines.append(f"- Justificación: {contexto['justificacion_gasto']}")

        # Información fiscal
        fiscal = self.extractor.extract_fiscal_info(factura_json)
        lines.append(f"\nINFORMACIÓN FISCAL:")
        lines.append(f"- Régimen IVA: {fiscal['regimen_iva']}")
        lines.append(f"- Retención aplicada: {fiscal['retencion_aplicada']} ({fiscal['tipo_retencion']}%)")
        lines.append(f"- Operación intracomunitaria: {fiscal['operacion_intracomunitaria']}")

        # Relación comercial
        relacion = self.extractor.extract_relacion_comercial(factura_json)
        lines.append(f"\nRELACIÓN COMERCIAL:")
        lines.append(f"- Tipo: {relacion['tipo_relacion']}")
        lines.append(f"- Empresa vinculada: {relacion['empresa_vinculada']}")
        lines.append(f"- Operación vinculada: {relacion['operacion_vinculada']}")

        # Fechas
        fechas = self.extractor.safe_get(factura_json, 'fechas', default={})
        lines.append(f"\nFECHAS:")
        lines.append(f"- Emisión: {fechas.get('emision', 'N/A')}")
        lines.append(f"- Prestación servicio: {fechas.get('prestacion_servicio', 'N/A')}")
        lines.append(f"- Pago: {fechas.get('pago_efectivo', 'N/A')}")

        return '\n'.join(lines)

    def build_precedentes_context(self, consultas: List[Dict]) -> str:
        """Construye el texto de contexto con los precedentes oficiales.

        Formatea las 3 consultas vinculantes más relevantes con su número,
        órgano, hechos resumidos y conclusiones clave para incluirlas
        en el prompt del modelo de lenguaje.

        Args:
            consultas: Lista de consultas tributarias relevantes.

        Returns:
            Cadena multilínea con los precedentes formateados, utilizando
            como máximo las 3 primeras consultas.
        """
        context_parts = []

        for i, consulta in enumerate(consultas[:3], 1):
            numero = consulta.get('numero_consulta', f'N/A-{i}')
            organo = consulta.get('organo', 'N/A')

            hechos = consulta.get('descripcion_hechos', '')[:300]
            resolucion = consulta.get('contestacion_completa', '')
            conclusiones = self.extract_key_conclusions_for_context(resolucion)

            context_parts.append(f"""
CONSULTA {numero} ({organo}):
Hechos: {hechos}...
Resolución clave: {conclusiones}
            """.strip())

        return '\n\n'.join(context_parts)

    def extract_key_conclusions_for_context(self, resolucion: str) -> str:
        """Extrae las conclusiones clave de una resolución para uso en contexto.

        Similar a ``extract_key_conclusions`` de ``ConsultasTributariasProcessor``,
        pero limitada a 2 oraciones y 400 caracteres para mantener el contexto
        del prompt compacto.

        Args:
            resolucion: Texto completo de la resolución/contestación oficial.

        Returns:
            Cadena con hasta 2 oraciones conclusivas, truncada a 400 caracteres.
        """
        keywords = [
            'se aplicará', 'resultará de aplicación', 'estará exenta',
            'será deducible', 'no se integrará', 'por tanto'
        ]

        sentences = resolucion.split('.')
        conclusions = []

        for sentence in sentences:
            sentence_clean = sentence.strip()
            if len(sentence_clean) > 20:
                sentence_lower = sentence_clean.lower()
                if any(keyword in sentence_lower for keyword in keywords):
                    conclusions.append(sentence_clean)
                    if len(conclusions) >= 2:
                        break

        return '. '.join(conclusions)[:400]

    def clean_json_response(self, response: str) -> str:
        """Limpia la respuesta cruda del VLLM para extraer un JSON válido.

        Localiza las llaves de apertura y cierre del objeto JSON y elimina
        caracteres de control que podrían invalidar el parsing.

        Args:
            response: Texto de respuesta completo del modelo de lenguaje.

        Returns:
            Cadena con el JSON limpio listo para ``json.loads()``.

        Raises:
            ValueError: Si no se encuentra un objeto JSON en la respuesta.
        """
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1

        if start_idx != -1 and end_idx != 0:
            json_str = response[start_idx:end_idx]
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
            json_str = re.sub(r'\\n', ' ', json_str)
            json_str = re.sub(r'\\t', ' ', json_str)
            return json_str

        raise ValueError("No se encontró JSON válido en respuesta VLLM")

    def enrich_recommendation(self, recommendation: Dict, factura_json: Dict,
                              consultas: List[Dict]) -> Dict:
        """Enriquece la recomendación con metadatos, ahorros y factores de riesgo.

        Agrega al diccionario de recomendación:
            - Metadatos del análisis (timestamp, consultas analizadas, importes).
            - Cálculo del ahorro fiscal total estimado sumando las oportunidades.
            - Lista de factores de riesgo detectados (operaciones vinculadas,
              intracomunitarias, retenciones no aplicadas).

        Args:
            recommendation: Diccionario de recomendación generada por VLLM.
            factura_json: Diccionario con los datos completos de la factura.
            consultas: Lista de consultas tributarias utilizadas en el análisis.

        Returns:
            El mismo diccionario ``recommendation`` enriquecido con las claves
            ``metadata``, ``ahorro_total_estimado`` y ``factores_riesgo``.
        """

        # Extraer información enriquecida
        importes = self.extractor.safe_get(factura_json, 'importes', default={})
        contexto = self.extractor.extract_contexto_empresarial(factura_json)

        # Añadir metadatos enriquecidos
        recommendation['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'consultas_analizadas': len(consultas),
            'consultas_top': [c.get('numero_consulta', 'N/A') for c in consultas[:3]],
            'importe_factura': importes.get('total_factura', '0'),
            'base_imponible': importes.get('base_imponible', '0'),
            'departamento_origen': contexto['departamento'],
            'proyecto_asociado': contexto['proyecto'],
            'afectacion_empresarial': f"{contexto['porcentaje_afectacion']}%"
        }

        # Calcular ahorros totales estimados
        total_savings = 0
        for opp in recommendation.get('oportunidades_fiscales', []):
            if 'ahorro_estimado_euros' in opp and isinstance(opp['ahorro_estimado_euros'], (int, float)):
                total_savings += opp['ahorro_estimado_euros']

        recommendation['ahorro_total_estimado'] = total_savings

        # Añadir contexto de riesgo fiscal
        relacion = self.extractor.extract_relacion_comercial(factura_json)
        fiscal = self.extractor.extract_fiscal_info(factura_json)

        risk_factors = []
        if relacion['operacion_vinculada']:
            risk_factors.append("Operación con empresa vinculada")
        if fiscal['operacion_intracomunitaria']:
            risk_factors.append("Operación intracomunitaria")
        if not fiscal['retencion_aplicada'] and importes.get('total_factura', 0) > 300:
            risk_factors.append("Posible retención no aplicada")

        recommendation['factores_riesgo'] = risk_factors

        return recommendation

    def fallback_recommendation(self) -> Dict:
        """Genera una recomendación de respaldo cuando el análisis principal falla.

        Devuelve una estructura completa con valores conservadores:
        código G19 (servicios profesionales) con confianza baja (0.3),
        una alerta de nivel alto indicando que se requiere revisión manual
        y sin oportunidades fiscales.

        Returns:
            Diccionario con la estructura completa de recomendación fiscal
            en modo fallback, incluyendo una alerta de sistema.
        """
        return {
            "clasificacion": {
                "codigo_principal": "G19",
                "codigo_alternativo": ["G42"],
                "confianza": 0.3,
                "justificacion": "Clasificación de fallback - requiere revisión manual"
            },
            "deducibilidad": {
                "es_deducible": True,
                "porcentaje_deducible": "100%",
                "condiciones_cumplimiento": ["revisar vinculación actividad"],
                "documentacion_requerida": ["factura completa", "justificación empresarial"],
                "verificaciones_adicionales": ["validar clasificación manual"]
            },
            "oportunidades_fiscales": [],
            "alertas_cumplimiento": [
                {
                    "tipo": "sistema",
                    "descripcion": "Error en análisis automático",
                    "nivel_riesgo": "alto",
                    "accion_requerida": "revisión manual urgente",
                    "plazo_accion": "inmediato"
                }
            ],
            "analisis_relacion_comercial": {
                "riesgo_vinculacion": "desconocido",
                "recomendaciones": ["verificar manualmente"]
            },
            "consultas_vinculantes_aplicables": [],
            "confidence_score": 0.3,
            "ahorro_total_estimado": 0,
            "factores_riesgo": ["análisis automático fallido"],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "status": "fallback"
            }
        }


class EnhancedFiscalClassifier:
    """Orquestador principal del sistema de clasificación fiscal v2.0.

    Integra todos los componentes del sistema:
        - ``ConsultasTributariasProcessor``: para procesar consultas tributarias.
        - ``ConsultasRecommendationEngine``: para generar recomendaciones.
        - ``FacturaDataExtractor``: para acceso seguro a datos de factura.
        - ``ClasificacionesFiscalesAEAT``: para enriquecer con modelos de
          presentación por código AEAT.

    Proporciona la interfaz de alto nivel para clasificar gastos, analizar
    completitud de datos y obtener estadísticas del sistema.

    Atributos:
        extractor: Instancia de ``FacturaDataExtractor``.
        consultas_processor: Procesador de consultas tributarias.
        processed_consultas: Lista de consultas procesadas con embeddings
            (``None`` hasta que se invoque ``load_consultas``).
        recommendation_engine: Motor de recomendaciones inicializado con
            las consultas procesadas (``None`` hasta que se invoque
            ``load_consultas``).
    """

    def __init__(self, consultas_json_file: str=None):
        """Inicializa el sistema de clasificación fiscal.

        Crea las instancias base del extractor y procesador. Si se proporciona
        un archivo o estructura de consultas, las carga y procesa
        inmediatamente.

        Args:
            consultas_json_file: Ruta al archivo JSON con las consultas
                tributarias, una lista de consultas, o un diccionario con
                las claves ``consultas_vinculantes`` y/o
                ``consultas_adicionales``. Si es ``None``, el sistema se
                inicializa sin consultas y se deberá invocar
                ``load_consultas`` posteriormente.
        """
        logger.info("Inicializando sistema de clasificación fiscal v2.0...")

        self.extractor = FacturaDataExtractor()

        # Procesar consultas
        self.consultas_processor = ConsultasTributariasProcessor()
        self.processed_consultas = None
        self.recommendation_engine = None

        # Cargar consultas desde archivo
        if consultas_json_file:
            self.load_consultas(consultas_json_file)


        logger.info("Sistema v2.0 inicializado correctamente")

    def load_consultas(self, json_file: Any):
        """Carga consultas tributarias desde diversas fuentes y las procesa.

        Acepta tres formatos de entrada:
            - ``list``: Lista directa de diccionarios de consultas.
            - ``dict``: Diccionario con claves ``consultas_vinculantes`` y/o
              ``consultas_adicionales``.
            - ``str``: Ruta a un archivo JSON que se lee desde disco.

        Tras cargar las consultas, las procesa con ``ConsultasTributariasProcessor``
        (generando embeddings y análisis fiscal) e inicializa el motor de
        recomendaciones.

        Args:
            json_file: Fuente de consultas (lista, diccionario o ruta de archivo).

        Raises:
            FileNotFoundError: Si la ruta de archivo no existe.
            json.JSONDecodeError: Si el archivo no contiene JSON válido.
        """
        if isinstance(json_file, list):
            self.raw_consultas = json_file
        elif isinstance(json_file, dict):
            consultas_principales = json_file.get('consultas_vinculantes') or []
            consultas_adicionales = json_file.get('consultas_adicionales') or []

            if not isinstance(consultas_principales, list):
                consultas_principales = []
            if not isinstance(consultas_adicionales, list):
                consultas_adicionales = []

            self.raw_consultas = consultas_principales + consultas_adicionales
            logger.info(
                "Consultas cargadas desde diccionario normalizado: %s principales, %s adicionales",
                len(consultas_principales),
                len(consultas_adicionales),
            )
        else:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.raw_consultas = json.load(f)

                logger.info(f"Cargadas {len(self.raw_consultas)} consultas desde {json_file}")

            except FileNotFoundError:
                logger.error(f"Archivo no encontrado: {json_file}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Error parseando JSON: {e}")
                raise
        # Procesar consultas
        logger.info("Procesando consultas tributarias...")
        self.consultas_processor = ConsultasTributariasProcessor()
        self.processed_consultas = self.consultas_processor.process_consultas_batch(
            self.raw_consultas
        )

        # Inicializar motor de recomendaciones
        logger.info("Inicializando motor de recomendaciones...")
        self.recommendation_engine = ConsultasRecommendationEngine(
            self.processed_consultas
        )

    def classify_expense_with_precedents(self, factura_json: Dict) -> Dict:
        """Clasifica un gasto empresarial usando precedentes oficiales de la AEAT.

        Flujo principal del sistema v2.0:
            1. Valida la estructura de la factura de entrada.
            2. Genera la recomendación fiscal mediante el motor de recomendaciones.
            3. Enriquece la clasificación con los modelos de presentación
               (formularios tributarios) correspondientes a cada código AEAT
               detectado, consultando ``ClasificacionesFiscalesAEAT``.

        Args:
            factura_json: Diccionario con los datos completos de la factura.
                Debe contener al menos la sección ``conceptos`` con al menos
                un concepto con descripción.

        Returns:
            Diccionario con la recomendación fiscal completa. La sección
            ``clasificacion`` incluye adicionalmente el campo
            ``presentaciones`` con los modelos tributarios aplicables
            por cada código AEAT.

        Raises:
            ValueError: Si la factura no cumple las validaciones mínimas
                (propagado desde ``validate_enhanced_factura_input``).
        """
        logger.info("Iniciando clasificación con precedentes oficiales v2.0...")

        try:
            # Validar entrada con nueva estructura
            self.validate_enhanced_factura_input(factura_json)

            # Generar recomendación usando estructura completa
            recommendation = self.recommendation_engine.get_best_recommendation(factura_json)

            # Enriquecer con presentaciones (modelos) por código G a partir de clasificaciones_fiscales_aeat.json
            try:
                fiscal_loader = ClasificacionesFiscalesAEAT("data/fiscal/clasificaciones_fiscales_aeat.json")

                clasif = recommendation.get('clasificacion', {}) or {}
                codigo_principal = clasif.get('codigo_principal')
                codigos_alternativos = clasif.get('codigo_alternativo', []) or []

                # Determinar régimen objetivo según tipo de empresa del receptor
                receptor_info = self.extractor.extract_empresa_info(factura_json, 'receptor')
                tipo_empr = (receptor_info.get('tipo_empresa') or '').strip().lower()
                regimen_objetivo = 'autonomo' if 'autonomo' in tipo_empr else 'sociedad'

                # Asegurar lista de códigos sin duplicados y válidos
                codigos: List[str] = []
                if isinstance(codigo_principal, str) and codigo_principal:
                    codigos.append(codigo_principal.upper())
                if isinstance(codigos_alternativos, list):
                    codigos.extend([str(c).upper() for c in codigos_alternativos if isinstance(c, str) and c])
                elif isinstance(codigos_alternativos, str) and codigos_alternativos:
                    codigos.append(codigos_alternativos.upper())
                codigos = list(dict.fromkeys(codigos))  # unique, keep order

                presentaciones: Dict[str, List[Dict[str, str]]] = {}

                for codigo in codigos:
                    data = fiscal_loader.get_clasificacion(codigo) or {}
                    regimenes = data.get('regimenes', {}) or {}

                    modelos_list: List[Dict[str, str]] = []
                    vistos = set()

                    def agregar_modelos(desde_regimenes: Dict[str, Dict[str, dict]]):
                        """Agrega modelos de presentación de los regimenes dados a ``modelos_list``.

                        Itera sobre los regimenes fiscales proporcionados, extrae sus
                        modelos tributarios y los acumula en la lista externa
                        ``modelos_list``, evitando duplicados mediante el conjunto
                        ``vistos``.

                        Args:
                            desde_regimenes: Diccionario de regimenes fiscales, donde
                                cada valor contiene un sub-diccionario ``modelos``.
                        """
                        for reg_data in (desde_regimenes or {}).values():
                            modelos = (reg_data or {}).get('modelos', {}) or {}
                            for modelo_cod, modelo_info in modelos.items():
                                if modelo_cod in vistos:
                                    continue
                                vistos.add(modelo_cod)
                                desc = (modelo_info or {}).get('descripcion', '') or ''
                                det = (modelo_info or {}).get('detalle', '') or ''
                                obs = desc if det == '' else f"{desc}. {det}"
                                modelos_list.append({
                                    'modelo': str(modelo_cod),
                                    'obs': obs.strip()
                                })

                    if regimenes:
                        # Intentar usar solo el régimen objetivo
                        if regimen_objetivo in regimenes and (regimenes[regimen_objetivo] or {}).get('modelos'):
                            agregar_modelos({regimen_objetivo: regimenes[regimen_objetivo]})
                        else:
                            # Fallback: agregar todos los regímenes disponibles y deduplicar
                            agregar_modelos(regimenes)
                    else:
                        # Fallback del demo: no hay regimenes/modelos definidos
                        nota = data.get('nota') or data.get('descripcion') or 'Consultar normativa vigente'
                        modelos_list.append({'modelo': 'N/A', 'obs': str(nota)})

                    presentaciones[codigo] = modelos_list

                if clasif is not None:
                    clasif['presentaciones'] = presentaciones
                    recommendation['clasificacion'] = clasif
            except Exception as e:
                logger.warning(f"No se pudo enriquecer con presentaciones: {e}")

            logger.info("Clasificación v2.0 completada exitosamente")
            return recommendation

        except Exception as e:
            logger.error(f"Error en clasificación v2.0: {e}")
            return self.recommendation_engine.fallback_recommendation()

    def validate_enhanced_factura_input(self, factura_json: Dict):
        """Valida que la factura cumpla los requisitos mínimos de estructura.

        Verifica que exista la sección ``conceptos``, que contenga al menos
        un concepto y que al menos uno tenga una descripción no vacía.

        Args:
            factura_json: Diccionario con los datos de la factura a validar.

        Raises:
            ValueError: Si falta la sección ``conceptos``, si la lista de
                conceptos está vacía, o si ningún concepto tiene descripción.
        """
        # Campos obligatorios mínimos
        required_sections = ['conceptos']
        for section in required_sections:
            if section not in factura_json:
                raise ValueError(f"Sección requerida faltante: {section}")

        # Validar conceptos
        conceptos = self.extractor.extract_conceptos_info(factura_json)
        if not conceptos:
            raise ValueError("La factura debe tener al menos un concepto")

        # Validar que al menos tenga descripción
        if not any(c['descripcion'] for c in conceptos):
            raise ValueError("Al menos un concepto debe tener descripción")

    def analyze_factura_completeness(self, factura_json: Dict) -> Dict:
        """Analiza el grado de completitud de los datos de una factura.

        Evalúa la presencia de cada sección principal de la factura
        (identificación, emisor, receptor, conceptos, importes, fechas,
        contexto empresarial, fiscal y relación comercial), asignando un
        peso a cada una. Calcula un porcentaje de completitud y genera
        recomendaciones para mejorar la calidad de los datos.

        Args:
            factura_json: Diccionario con los datos de la factura a analizar.

        Returns:
            Diccionario con las claves:
                - ``score``: Puntuación obtenida.
                - ``max_score``: Puntuación máxima posible.
                - ``percentage``: Porcentaje de completitud.
                - ``missing_fields``: Lista de secciones faltantes.
                - ``recommendations``: Lista de sugerencias para mejorar
                  los datos.
        """
        completeness = {
            'score': 0,
            'max_score': 0,
            'missing_fields': [],
            'recommendations': []
        }

        # Verificar secciones principales
        sections_check = {
            'identificacion': 2,
            'emisor': 3,
            'receptor': 3,
            'conceptos': 4,
            'importes': 3,
            'fechas': 2,
            'contexto_empresarial': 3,
            'fiscal': 2,
            'relacion_comercial': 1
        }

        for section, weight in sections_check.items():
            completeness['max_score'] += weight
            if section in factura_json and factura_json[section]:
                completeness['score'] += weight
            else:
                completeness['missing_fields'].append(section)

        # Calcular porcentaje
        completeness['percentage'] = (completeness['score'] / completeness['max_score']) * 100

        # Generar recomendaciones
        if completeness['percentage'] < 60:
            completeness['recommendations'].append("Datos insuficientes para análisis preciso")
        if 'contexto_empresarial' in completeness['missing_fields']:
            completeness['recommendations'].append("Añadir contexto empresarial mejorará precisión")
        if 'fiscal' in completeness['missing_fields']:
            completeness['recommendations'].append("Información fiscal ayudará en alertas de cumplimiento")

        return completeness

    def get_enhanced_system_stats(self) -> Dict:
        """Devuelve estadísticas enriquecidas del estado actual del sistema.

        Incluye conteos de consultas cargadas/procesadas, códigos AEAT y
        CNAEs más frecuentes, y la lista de capacidades del sistema v2.0.

        Returns:
            Diccionario con las claves: ``version``, ``consultas_cargadas``,
            ``consultas_procesadas``, ``consultas_con_analisis``,
            ``codigos_detectados`` (top 10), ``cnaes_relevantes`` (top 10),
            ``timestamp`` y ``capabilities``.
        """
        return {
            "version": "2.0",
            "consultas_cargadas": len(self.raw_consultas),
            "consultas_procesadas": len(self.processed_consultas),
            "consultas_con_analisis": sum(1 for c in self.processed_consultas
                                          if c.get('fiscal_analysis')),
            "codigos_detectados": self._count_detected_codes(),
            "cnaes_relevantes": self._count_relevant_cnaes(),
            "timestamp": datetime.now().isoformat(),
            "capabilities": [
                "Estructura de datos completa",
                "Análisis contextual avanzado",
                "Detección operaciones vinculadas",
                "Análisis relación comercial",
                "Verificaciones fiscales específicas"
            ]
        }

    def _count_detected_codes(self) -> Dict:
        """Cuenta la frecuencia de códigos AEAT detectados en las consultas procesadas.

        Returns:
            Diccionario con los 10 códigos AEAT más frecuentes como claves
            y sus conteos como valores, ordenado de mayor a menor frecuencia.
        """
        code_count = {}
        for consulta in self.processed_consultas:
            fiscal_analysis = consulta.get('fiscal_analysis', {})
            for code in fiscal_analysis.get('codigos_aeat_aplicables', []):
                code_count[code] = code_count.get(code, 0) + 1
        return dict(sorted(code_count.items(), key=lambda x: x[1], reverse=True)[:10])

    def _count_relevant_cnaes(self) -> Dict:
        """Cuenta la frecuencia de códigos CNAE detectados en las consultas procesadas.

        Returns:
            Diccionario con los 10 CNAEs más frecuentes como claves
            y sus conteos como valores, ordenado de mayor a menor frecuencia.
        """
        cnae_count = {}
        for consulta in self.processed_consultas:
            fiscal_analysis = consulta.get('fiscal_analysis', {})
            for cnae in fiscal_analysis.get('cnaes_relevantes', []):
                cnae_count[cnae] = cnae_count.get(cnae, 0) + 1
        return dict(sorted(cnae_count.items(), key=lambda x: x[1], reverse=True)[:10])


# Ejemplo de uso con estructura completa
if __name__ == "__main__":
    inicio = time.perf_counter()
    # Inicializar sistema
    from aeat_scraper import AEATConsultaScraper
    scraper = AEATConsultaScraper(
        headless=True,
        verbose=True,
        use_pool=True,  # Habilitar pool de drivers
        pool_size=20  # Número de drivers en paralelo
    )

    factura_ejemplo = {
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
            "retencion_aplicada": True,
            "tipo_retencion": 15,
            "operacion_intracomunitaria": False
        },
        "relacion_comercial": {
            "tipo_relacion": "Tercero independiente",
            "empresa_vinculada": False,
            "operacion_vinculada": False,
            "proveedor_habitual": True
        },
        "fechas": {
            "emision": "2024-10-15",
            "prestacion_servicio": "2024-10-01",
            "pago_efectivo": "2024-10-20"
        }
    }
    results = scraper.search_comprehensive(factura_ejemplo)

    classifier = EnhancedFiscalClassifier(results)

    # Factura de ejemplo con estructura completa
    factura_json = factura_ejemplo
    # Analizar completitud de datos
    completeness = classifier.analyze_factura_completeness(factura_ejemplo)
    print("=== ANÁLISIS DE COMPLETITUD ===")
    print(json.dumps(completeness, indent=2, ensure_ascii=False))

    # Obtener recomendación
    resultado = classifier.classify_expense_with_precedents(factura_ejemplo)

    print("\n=== RECOMENDACIÓN FISCAL v2.0 ===")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))

    # Estadísticas del sistema
    stats = classifier.get_enhanced_system_stats()
    print("\n=== ESTADÍSTICAS DEL SISTEMA v2.0 ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    fin = time.perf_counter()
    duracion = fin - inicio
    print(f"\n⏱️ Tiempo total de ejecución: {duracion:.2f} segundos")
