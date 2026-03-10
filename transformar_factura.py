#!/usr/bin/env python3
"""
Script para transformar facturas de fg.json al formato mínimo requerido.
"""

import json
from typing import Dict, List, Any
from datetime import datetime


def transformar_factura(registro: Dict[str, Any], receptor_info: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Transforma un registro de factura al formato mínimo.

    Args:
        registro: Diccionario con datos de la factura original
        receptor_info: Información del receptor (opcional). Debe contener 'cif' y 'actividad_cnae'

    Returns:
        Diccionario con la factura en formato mínimo
    """
    # Extraer fecha de emisión y formatearla
    fecha_emision = registro.get("fecha_emision", "")
    # Si tiene formato datetime, convertir a solo fecha
    if " " in fecha_emision:
        fecha_emision = fecha_emision.split(" ")[0]

    # Construir concepto
    descripcion = registro.get("nombre_item", "")
    if not descripcion and registro.get("item_descripcion"):
        descripcion = registro.get("item_descripcion")

    conceptos = [{
        "descripcion": descripcion,
        "importe": float(registro.get("precio_unitario", 0))
    }]

    # Información del emisor
    emisor = {
        "nombre": registro.get("nombre_emisor", ""),
        "cif": registro.get("cif_emisor", "")
    }

    # Información del receptor (si se proporciona)
    if receptor_info is None:
        # Basado en análisis de 502 facturas y patrón de consumo:
        # - 109 gastos tecnológicos, nóminas desarrolladores/diseñadores
        # - 496 proveedores únicos, oficina física, operaciones complejas
        receptor_info = {
            "cif": "B12345678",  # PLACEHOLDER - CIF no deducible, requiere tabla companies (company_id=612)
            "actividad_cnae": "6201 - Programación informática"  # Deducido con ALTA confianza
        }

    receptor = {
        "actividad_cnae": receptor_info.get("actividad_cnae", ""),
        "cif": receptor_info.get("cif", "")
    }

    # Importes
    base_imponible = float(registro.get("base_imponible", 0))
    iva = float(registro.get("total_impuestos", 0))
    total = float(registro.get("total_factura", 0))

    importes = {
        "total": total,
        "base_imponible": base_imponible,
        "iva": iva
    }

    # Construir factura mínima
    factura_minima = {
        "conceptos": conceptos,
        "receptor": receptor,
        "emisor": emisor,
        "importes": importes,
        "fechas": {
            "emision": fecha_emision
        }
    }

    return factura_minima


def cargar_facturas(archivo: str = "data/facturas/fg.json") -> List[Dict[str, Any]]:
    """
    Carga las facturas del archivo JSON.

    Args:
        archivo: Ruta al archivo JSON

    Returns:
        Lista de registros de facturas
    """
    with open(archivo, 'r', encoding='utf-8') as f:
        contenido = json.load(f)

    # El archivo tiene estructura especial de PHPMyAdmin
    # Los datos están en el índice 2 bajo la clave 'data'
    if isinstance(contenido, list) and len(contenido) > 2:
        if isinstance(contenido[2], dict) and 'data' in contenido[2]:
            return contenido[2]['data']

    return contenido if isinstance(contenido, list) else []


def main():
    """Función principal para demostrar el uso."""

    # Cargar facturas
    print("Cargando facturas de fg.json...")
    facturas = cargar_facturas("data/facturas/fg.json")
    print(f"Se encontraron {len(facturas)} registros de factura\n")

    # Información del receptor (personalizar según necesidad)
    receptor_info = {
        "cif": "B87654321",
        "actividad_cnae": "6201 - Programación informática"
    }

    # Transformar la primera factura como ejemplo
    if facturas:
        print("=" * 70)
        print("EJEMPLO: Primera factura transformada")
        print("=" * 70)

        factura_original = facturas[0]
        print(f"\nFactura original: {factura_original.get('numero_factura')}")
        print(f"Emisor: {factura_original.get('nombre_emisor')}")
        print(f"Importe total: {factura_original.get('total_factura')} €")

        factura_transformada = transformar_factura(factura_original, receptor_info)

        print("\nFactura en formato mínimo:")
        print(json.dumps(factura_transformada, indent=2, ensure_ascii=False))

        # Opción: Guardar todas las facturas transformadas
        print("\n" + "=" * 70)
        respuesta = input("\n¿Deseas transformar TODAS las facturas y guardarlas? (s/n): ")

        if respuesta.lower() == 's':
            facturas_transformadas = []
            for factura in facturas:
                facturas_transformadas.append(transformar_factura(factura, receptor_info))

            archivo_salida = "data/facturas/facturas_transformadas.json"
            with open(archivo_salida, 'w', encoding='utf-8') as f:
                json.dump(facturas_transformadas, f, indent=2, ensure_ascii=False)

            print(f"\n✓ {len(facturas_transformadas)} facturas transformadas guardadas en: {archivo_salida}")
        else:
            print("\nTransformación cancelada.")
    else:
        print("No se encontraron facturas en el archivo.")


if __name__ == "__main__":
    main()
