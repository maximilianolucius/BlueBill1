#!/usr/bin/env python3
"""
Deducir características del receptor analizando QUÉ consume.
"""

import json
from collections import Counter
from typing import Dict, List, Any


def cargar_facturas(archivo: str = "data/facturas/fg.json") -> List[Dict[str, Any]]:
    """Carga las facturas del archivo JSON."""
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = json.load(f)

    if isinstance(contenido, list) and len(contenido) > 2:
        if isinstance(contenido[2], dict) and 'data' in contenido[2]:
            return contenido[2]['data']
    return []


def analizar_tipo_gastos(facturas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analiza los tipos de gastos para deducir la actividad real de la empresa.
    """

    # Recolectar todos los conceptos/items
    conceptos = []
    descripciones = []
    emisores = []

    for factura in facturas:
        nombre_item = factura.get('nombre_item', '') or ''
        item_descripcion = factura.get('item_descripcion', '') or ''
        nombre_emisor = factura.get('nombre_emisor', '') or ''

        if nombre_item:
            conceptos.append(nombre_item.lower())
        if item_descripcion:
            descripciones.append(item_descripcion.lower())
        if nombre_emisor:
            emisores.append(nombre_emisor)

    # Categorizar gastos
    categorias = {
        'tecnologia': [],
        'oficina': [],
        'servicios_profesionales': [],
        'suministros': [],
        'marketing': [],
        'formacion': [],
        'otros': []
    }

    keywords_tech = ['github', 'canva', 'software', 'servidor', 'hosting', 'cloud', 'api',
                     'licencia', 'suscripción', 'dominio', 'ssl', 'desarrollo', 'programación',
                     'informática', 'portátil', 'ordenador', 'mac', 'dell', 'hp', 'microsoft']

    keywords_oficina = ['alquiler', 'oficina', 'espacio', 'coworking', 'mobiliario']

    keywords_servicios = ['consultoría', 'asesoría', 'gestoría', 'abogado', 'notario',
                          'registro', 'auditoría', 'contabilidad']

    keywords_suministros = ['eléctrico', 'agua', 'gas', 'internet', 'telefonía',
                           'luz', 'residuos', 'tasa']

    keywords_marketing = ['publicidad', 'marketing', 'diseño', 'gráfico', 'social media',
                         'seo', 'ads', 'campaña']

    keywords_formacion = ['formación', 'curso', 'certificación', 'training', 'bootcamp',
                         'conferencia', 'evento', 'colegio profesional', 'cuota colegio']

    all_texts = conceptos + descripciones

    for texto in all_texts:
        categorizado = False

        if any(kw in texto for kw in keywords_tech):
            categorias['tecnologia'].append(texto)
            categorizado = True
        if any(kw in texto for kw in keywords_oficina):
            categorias['oficina'].append(texto)
            categorizado = True
        if any(kw in texto for kw in keywords_servicios):
            categorias['servicios_profesionales'].append(texto)
            categorizado = True
        if any(kw in texto for kw in keywords_suministros):
            categorias['suministros'].append(texto)
            categorizado = True
        if any(kw in texto for kw in keywords_marketing):
            categorias['marketing'].append(texto)
            categorizado = True
        if any(kw in texto for kw in keywords_formacion):
            categorias['formacion'].append(texto)
            categorizado = True

        if not categorizado and texto:
            categorias['otros'].append(texto)

    return {
        'categorias': categorias,
        'total_conceptos': len(all_texts),
        'emisores_unicos': len(set(emisores)),
        'top_emisores': Counter(emisores).most_common(10)
    }


def deducir_cnae_por_consumo(analisis: Dict[str, Any]) -> List[str]:
    """
    Deduce posibles CNAEs basándose en el patrón de consumo.
    """
    categorias = analisis['categorias']

    cnaes_probables = []

    # Si tiene muchos gastos en tecnología
    tech_ratio = len(categorias['tecnologia']) / max(analisis['total_conceptos'], 1)

    if tech_ratio > 0.3:
        cnaes_probables.append({
            'cnae': '6201',
            'descripcion': 'Programación informática',
            'confianza': 'ALTA',
            'razon': f'{len(categorias["tecnologia"])} gastos tecnológicos ({tech_ratio*100:.1f}% del total)'
        })

    if tech_ratio > 0.2:
        cnaes_probables.append({
            'cnae': '6202',
            'descripcion': 'Consultoría informática',
            'confianza': 'MEDIA-ALTA',
            'razon': 'Perfil compatible con servicios IT'
        })

    # Si tiene gastos de oficina
    if len(categorias['oficina']) > 0:
        cnaes_probables.append({
            'cnae': '62XX',
            'descripcion': 'Servicios de TI con oficina física',
            'confianza': 'MEDIA',
            'razon': f'{len(categorias["oficina"])} gastos de oficina/alquiler'
        })

    return cnaes_probables


def estimar_tipo_sociedad(analisis: Dict[str, Any]) -> str:
    """
    Estima el tipo de sociedad (forma jurídica) basándose en patrones.
    """
    # Por el número de emisores y volumen, podemos hacer estimaciones
    if analisis['emisores_unicos'] > 50:
        return "Probablemente SL o SA (volumen medio-alto de operaciones)"
    elif analisis['emisores_unicos'] > 20:
        return "Probablemente SL (pequeña-mediana empresa)"
    else:
        return "Posiblemente Autónomo o microempresa"


def main():
    print("=" * 70)
    print("ANÁLISIS DEL RECEPTOR POR PATRÓN DE CONSUMO")
    print("=" * 70)

    facturas = cargar_facturas("fg.json")
    print(f"\nTotal facturas analizadas: {len(facturas)}")

    print("\n" + "=" * 70)
    print("1. ANÁLISIS DE GASTOS POR CATEGORÍA")
    print("=" * 70)

    analisis = analizar_tipo_gastos(facturas)

    for categoria, items in analisis['categorias'].items():
        if items:
            print(f"\n{categoria.upper()}: {len(items)} gastos")
            # Mostrar algunos ejemplos únicos
            ejemplos = list(set(items))[:5]
            for ejemplo in ejemplos:
                print(f"  - {ejemplo[:60]}{'...' if len(ejemplo) > 60 else ''}")

    print("\n" + "=" * 70)
    print("2. PRINCIPALES PROVEEDORES")
    print("=" * 70)

    for emisor, count in analisis['top_emisores'][:10]:
        if emisor:
            print(f"  {count:3d} facturas - {emisor}")

    print("\n" + "=" * 70)
    print("3. DEDUCCIÓN DE CNAE")
    print("=" * 70)

    cnaes_deducidos = deducir_cnae_por_consumo(analisis)

    for cnae_info in cnaes_deducidos:
        print(f"\n  CNAE: {cnae_info['cnae']}")
        print(f"  Descripción: {cnae_info['descripcion']}")
        print(f"  Confianza: {cnae_info['confianza']}")
        print(f"  Razón: {cnae_info['razon']}")

    print("\n" + "=" * 70)
    print("4. TIPO DE EMPRESA (ESTIMACIÓN)")
    print("=" * 70)

    tipo = estimar_tipo_sociedad(analisis)
    print(f"\n  {tipo}")
    print(f"  Emisores únicos: {analisis['emisores_unicos']}")

    print("\n" + "=" * 70)
    print("5. PERFIL DEDUCIDO")
    print("=" * 70)

    tech_count = len(analisis['categorias']['tecnologia'])
    total = analisis['total_conceptos']

    print(f"""
  La empresa receptora es muy probablemente:

  - Actividad: Empresa de tecnología / Programación informática
  - CNAE más probable: 6201 (Programación informática)
  - Perfil: {tech_count}/{total} gastos ({tech_count/max(total,1)*100:.1f}%) relacionados con tecnología
  - Tamaño: {analisis['emisores_unicos']} proveedores diferentes
  - Tipo sociedad: Probablemente SL (Sociedad Limitada)

  CIF: No se puede deducir (es un identificador único asignado)
       Formato español típico: B + 8 dígitos (para SL)
       Ejemplo: B12345678
    """)


if __name__ == "__main__":
    main()
