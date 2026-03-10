"""
Generador de palabras clave y términos fiscales a partir de datos de facturas.

Módulo utilitario que analiza los datos estructurados de una factura y genera
dos listas: palabras clave tributarias relevantes (keywords) y términos fiscales
y contables (fiscal_terms). Utiliza el modelo de lenguaje (LangChainVLLMAdapter)
para generar las listas de forma inteligente a partir del contexto fiscal de
la factura. En caso de que el modelo no esté disponible o la respuesta no sea
parseable, recurre a un mecanismo heurístico de respaldo que extrae términos
directamente de los campos de la factura.
"""

import json
from typing import Tuple, List, Dict, Any
from utils.generator_model import LangChainVLLMAdapter
from typing import Tuple, List, Dict, Any

def _heuristic_fallback(factura_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Genera listas básicas de keywords y términos fiscales por heurística.

    Mecanismo de respaldo robusto que se activa cuando el modelo vLLM no
    está disponible o produce respuestas no parseables. Extrae términos
    directamente de los campos de la factura como conceptos, tipo de IVA,
    CNAE del emisor/receptor, régimen de IVA y retenciones.

    Args:
        factura_data: Diccionario con los datos estructurados de la factura.
            Se esperan claves como 'conceptos', 'receptor', 'emisor',
            'fiscal' e 'importes'.

    Returns:
        Tupla con dos listas deduplicadas y limitadas a 64 elementos:
            - keywords: Palabras clave tributarias y fiscales básicas.
            - fiscal_terms: Términos clave fiscales y contables.
    """
    keywords: List[str] = []
    fiscal_terms: List[str] = []


    print("_heuristic_fallback <---< ")
    try:
        conceptos = factura_data.get('conceptos', [])
        desc = conceptos[0].get('descripcion', '') if conceptos else ''
        tipo_iva = conceptos[0].get('tipo_iva') if conceptos else factura_data.get('importes', {}).get('tipo_iva')
        receptor_cnae = factura_data.get('receptor', {}).get('actividad_cnae', '')
        emisor_cnae = factura_data.get('emisor', {}).get('actividad_cnae', '')
        regimen = factura_data.get('fiscal', {}).get('regimen_iva', '')
        ret_apl = factura_data.get('fiscal', {}).get('retencion_aplicada', False)
        tipo_ret = factura_data.get('fiscal', {}).get('tipo_retencion')

        base_keywords = ["deducible", "gasto", "IVA", "IRPF", "servicios profesionales", "consultoría", "informática"]
        keywords.extend(base_keywords)
        if desc:
            for token in desc.split():
                token = token.strip().lower().strip(',.;')
                if token and token not in keywords and len(token) > 3:
                    keywords.append(token)
        if receptor_cnae:
            keywords.append(receptor_cnae.split(' - ')[0])
        if emisor_cnae:
            keywords.append(emisor_cnae.split(' - ')[0])

        if tipo_iva is not None:
            fiscal_terms.append(f"IVA {tipo_iva}%")
        if regimen:
            fiscal_terms.append(f"Régimen {regimen}")
        if ret_apl and tipo_ret is not None:
            fiscal_terms.append(f"IRPF retención {tipo_ret}%")
        if desc:
            fiscal_terms.append(desc)
        if receptor_cnae:
            fiscal_terms.append(f"CNAE {receptor_cnae}")
        if emisor_cnae:
            fiscal_terms.append(f"CNAE {emisor_cnae}")
    except Exception:
        pass

    # Desduplicar preservando orden
    def dedup(seq: List[str]) -> List[str]:
        """Elimina duplicados de una lista preservando el orden original.

        Args:
            seq: Lista de cadenas posiblemente con elementos repetidos.

        Returns:
            Lista con elementos únicos en el orden de primera aparición.
        """
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedup(keywords)[:64], dedup(fiscal_terms)[:64]

def generar_keywords_y_fiscal_terms(
    factura_data: Dict[str, Any],
    max_tokens: int = 10_000,
    temperature: float = 0.5
) -> Tuple[List[str], List[str]]:
    """Genera keywords relevantes y términos fiscales a partir de datos de una factura.

    Construye un prompt especializado con la información fiscal de la factura
    (concepto, IVA, CNAE, régimen, retenciones) y lo envía al modelo
    LangChainVLLMAdapter para obtener listas inteligentes en formato JSON.
    Si el modelo no está disponible o la respuesta no se puede parsear,
    recurre automáticamente al mecanismo heurístico (_heuristic_fallback).

    Args:
        factura_data: Diccionario con los datos estructurados de la factura.
            Claves esperadas: 'conceptos' (lista con 'descripcion' y 'tipo_iva'),
            'receptor' y 'emisor' (con 'actividad_cnae'), 'fiscal' (con
            'regimen_iva', 'retencion_aplicada', 'tipo_retencion').
        max_tokens: Número máximo de tokens para la generación del modelo.
            Por defecto 10000.
        temperature: Parámetro de temperatura que controla la creatividad
            de la generación. Por defecto 0.5.

    Returns:
        Tupla con dos listas:
            - keywords_relevantes: Palabras clave tributarias y fiscales
              asociadas a la factura.
            - fiscal_terms: Combinaciones o frases clave relevantes para
              temas fiscales y contables.
    """

    # Extraer datos de forma segura
    concepto_desc = factura_data.get('conceptos', [{}])[0].get('descripcion', 'N/A')
    tipo_iva = factura_data.get('conceptos', [{}])[0].get('tipo_iva', 'N/A')
    receptor_cnae = factura_data.get('receptor', {}).get('actividad_cnae', 'N/A')
    emisor_cnae = factura_data.get('emisor', {}).get('actividad_cnae', 'N/A')
    regimen_iva = factura_data.get('fiscal', {}).get('regimen_iva', 'N/A')
    retencion_aplicada = factura_data.get('fiscal', {}).get('retencion_aplicada', False)
    tipo_retencion = factura_data.get('fiscal', {}).get('tipo_retencion', 'N/A')

    prompt = f"""
    Eres un asistente experto en temas tributarios. Dado el siguiente resumen de una factura con información fiscal y económica:

    Concepto de la factura: {concepto_desc}
    Tipo de IVA aplicado: {tipo_iva}%
    Actividad económica del receptor: {receptor_cnae}
    Actividad económica del emisor: {emisor_cnae}
    Régimen de IVA: {regimen_iva}
    ¿Se aplica retención? {retencion_aplicada}
    Tipo de retención aplicada: {tipo_retencion}%

    De acuerdo con esta información, genera dos listas en formato JSON:

    1. keywords_relevantes: palabras clave tributarias y fiscales relevantes asociadas a esta factura (ej.: deducible, gasto, IVA, IRPF, servicios profesionales, etc.)

    2. fiscal_terms: combinaciones o frases claves relevantes para temas fiscales y contables que se puedan usar para búsquedas y análisis.

    ---
    Ejemplo de formato de salida:

    {{
      "keywords_relevantes": [...],
      "fiscal_terms": [...]
    }}

    IMPORTANTE: escribe la salida estrictamente en formato JSON válido, sin texto adicional fuera del JSON.
    """

    def pre_parse_json(response: str) -> str:
        """Extrae el bloque JSON de una respuesta de texto del modelo.

        Busca la primera llave de apertura y la última de cierre para
        aislar el objeto JSON de cualquier texto adicional.

        Args:
            response: Texto completo de la respuesta del modelo.

        Returns:
            Cadena con el bloque JSON extraído.

        Raises:
            ValueError: Si no se encuentra un bloque JSON válido en la respuesta.
        """
        start = response.find('{')
        end = response.rfind('}')
        if start == -1 or end == -1:
            raise ValueError("No se encontró un bloque JSON en la respuesta")
        return response[start:end+1]

    try:
        vllm = LangChainVLLMAdapter(max_tokens=max_tokens, temperature=temperature)
        respuesta_raw = vllm.generate(prompt)
        if '</think>' in respuesta_raw:
            respuesta_raw = respuesta_raw.split('</think')[-1]

    except Exception as e:
        # Degradar con heurística si no hay vLLM
        return _heuristic_fallback(factura_data)

    try:
        if "</think>" in respuesta_raw:
            respuesta_raw = respuesta_raw.split('</think>')[-1]
        respuesta_json_str = pre_parse_json(respuesta_raw)
        datos_generados = json.loads(respuesta_json_str)
        keywords_relevantes = datos_generados.get("keywords_relevantes", [])
        fiscal_terms = datos_generados.get("fiscal_terms", [])
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error al parsear la respuesta JSON: {e}")
        print(f"Respuesta útil del modelo:\n{respuesta_raw}")
        # Si el JSON no es parseable, usar fallback
        keywords_relevantes, fiscal_terms = _heuristic_fallback(factura_data)

    return keywords_relevantes, fiscal_terms


if __name__ == '__main__':
    # Ejemplo factura_data (en realidad puede ser dinámica)
    factura_data = {
        "conceptos": [
            {
                "descripcion": "Servicios consultoría informática desarrollo ERP",
                "tipo_iva": 21
            }
        ],
        "receptor": {
            "actividad_cnae": "6201 - Programación informática"
        },
        "emisor": {
            "actividad_cnae": "6202 - Consultoría informática"
        },
        "fiscal": {
            "regimen_iva": "General",
            "retencion_aplicada": True,
            "tipo_retencion": 15
        }
    }

    keywords, fisc_terms = generar_keywords_y_fiscal_terms(factura_data)
    print("Keywords Relevantes:", keywords)
    print("Fiscal Terms:", fisc_terms)
