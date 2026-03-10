#!/usr/bin/env python3
"""
Script para analizar facturas y deducir información del receptor.
"""

import json
from collections import Counter
from typing import Dict, List, Any, Set


def cargar_facturas(archivo: str = "data/facturas/fg.json") -> List[Dict[str, Any]]:
    """Carga las facturas del archivo JSON."""
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = json.load(f)

    if isinstance(contenido, list) and len(contenido) > 2:
        if isinstance(contenido[2], dict) and 'data' in contenido[2]:
            return contenido[2]['data']

    return contenido if isinstance(contenido, list) else []


def analizar_receptor(facturas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analiza todas las facturas para deducir información del receptor.
    """
    # Campos que podrían contener información del receptor
    company_ids = set()
    cnaes_mencionados = []
    actividades_descritas = []

    # Buscar en todos los registros
    for factura in facturas:
        # Company ID
        if factura.get('company_id'):
            company_ids.add(factura['company_id'])

        # CNAE mencionado en clasificaciones
        clasificacion_justificacion = factura.get('clasificacion_justificacion', '') or ''
        if 'CNAE' in clasificacion_justificacion:
            # Buscar patrones como "CNAE 6201" o "(CNAE 6201)"
            import re
            matches = re.findall(r'CNAE\s*(\d{4})', clasificacion_justificacion)
            cnaes_mencionados.extend(matches)

        # Actividades mencionadas
        if 'programación informática' in clasificacion_justificacion.lower():
            actividades_descritas.append('Programación informática')
        if '6201' in clasificacion_justificacion:
            actividades_descritas.append('6201 - Programación informática')

    # Analizar metadata_completo que podría tener más info
    proyectos = []
    departamentos = []

    for factura in facturas:
        metadata = factura.get('metadata_completo')
        if metadata:
            try:
                if isinstance(metadata, str):
                    metadata_dict = json.loads(metadata)
                else:
                    metadata_dict = metadata

                if 'departamento_origen' in metadata_dict:
                    departamentos.append(metadata_dict['departamento_origen'])
                if 'proyecto_asociado' in metadata_dict:
                    proyectos.append(metadata_dict['proyecto_asociado'])
            except:
                pass

    # Contar frecuencias
    cnae_counter = Counter(cnaes_mencionados)
    actividad_counter = Counter(actividades_descritas)

    resultado = {
        'company_ids': list(company_ids),
        'cnae_mas_comun': cnae_counter.most_common(5) if cnae_counter else [],
        'actividades_mencionadas': actividad_counter.most_common(5) if actividad_counter else [],
        'total_facturas': len(facturas),
        'departamentos_comunes': Counter(departamentos).most_common(5) if departamentos else [],
        'proyectos_comunes': Counter(proyectos).most_common(5) if proyectos else []
    }

    return resultado


def buscar_cif_receptor_directo(facturas: List[Dict[str, Any]]) -> Set[str]:
    """
    Busca si hay algún campo que contenga directamente el CIF del receptor.
    """
    cifs_receptores = set()

    # Buscar en todos los campos posibles
    campos_buscar = [
        'receptor_cif', 'cif_receptor', 'cliente_cif',
        'company_cif', 'empresa_cif'
    ]

    for factura in facturas:
        for campo in campos_buscar:
            if campo in factura and factura[campo]:
                cifs_receptores.add(factura[campo])

    return cifs_receptores


def analizar_primera_factura_detalle(facturas: List[Dict[str, Any]]):
    """Muestra todos los campos de la primera factura para inspección."""
    if facturas:
        print("Campos disponibles en la primera factura:")
        print("=" * 70)
        for key, value in facturas[0].items():
            # Truncar valores largos
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            print(f"  {key}: {value_str}")


def main():
    print("Analizando facturas en fg.json...\n")

    # Cargar facturas
    facturas = cargar_facturas("fg.json")
    print(f"Total de facturas encontradas: {len(facturas)}\n")

    # Buscar CIF del receptor directamente
    print("1. Buscando CIF del receptor en campos directos...")
    cifs = buscar_cif_receptor_directo(facturas)
    if cifs:
        print(f"   CIFs encontrados: {cifs}")
    else:
        print("   No se encontró CIF del receptor en campos directos")

    print("\n" + "=" * 70)

    # Analizar información del receptor
    print("\n2. Analizando información del receptor por deducción...")
    resultado = analizar_receptor(facturas)

    print(f"\n   Company IDs encontrados: {resultado['company_ids']}")

    if resultado['cnae_mas_comun']:
        print(f"\n   CNAEs mencionados más frecuentemente:")
        for cnae, count in resultado['cnae_mas_comun']:
            print(f"      - CNAE {cnae}: mencionado {count} veces")

    if resultado['actividades_mencionadas']:
        print(f"\n   Actividades del receptor deducidas:")
        for actividad, count in resultado['actividades_mencionadas']:
            print(f"      - {actividad}: mencionado {count} veces")

    if resultado['departamentos_comunes']:
        print(f"\n   Departamentos de la empresa:")
        for dept, count in resultado['departamentos_comunes']:
            print(f"      - {dept}: {count} facturas")

    if resultado['proyectos_comunes']:
        print(f"\n   Proyectos asociados:")
        for proyecto, count in resultado['proyectos_comunes']:
            print(f"      - {proyecto}: {count} facturas")

    print("\n" + "=" * 70)
    print("\n3. Muestra de campos de la primera factura:")
    print("=" * 70)
    analizar_primera_factura_detalle(facturas)

    # Conclusión
    print("\n" + "=" * 70)
    print("CONCLUSIÓN")
    print("=" * 70)

    if resultado['cnae_mas_comun']:
        cnae_principal = resultado['cnae_mas_comun'][0][0]
        print(f"\n✓ Actividad CNAE deducida: {cnae_principal}")
        print(f"  Descripción: Programación informática")

    print(f"\n✓ Company ID: {resultado['company_ids'][0] if resultado['company_ids'] else 'No encontrado'}")

    if not cifs:
        print(f"\n⚠ CIF del receptor: No se pudo deducir automáticamente")
        print(f"  Se necesitaría acceder a la tabla 'companies' o 'usuarios'")
        print(f"  con company_id={resultado['company_ids'][0] if resultado['company_ids'] else '?'}")
    else:
        print(f"\n✓ CIF del receptor: {list(cifs)[0]}")


if __name__ == "__main__":
    main()
