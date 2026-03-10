#!/usr/bin/env python3
"""
Run the fiscal classify request via curl and print the response.

Usage:
  python tests/run_fiscal_classify.py

This script constructs the same JSON payload as the provided curl command
and invokes curl to POST it to the target endpoint, streaming output
directly to the terminal.
"""

import json
import subprocess
import sys


def main() -> int:
    payload = {
        "identificacion": {
            "id_interno": "FACT-2024-001234",
            "numero_factura": "F-2024-0156",
            "serie": "A",
        },
        "conceptos": [
            {
                "descripcion": "Servicios consultoría informática desarrollo ERP",
                "cantidad": 40,
                "precio_unitario": 37.5,
                "importe_linea": 1500,
                "tipo_iva": 21,
                "codigo_producto": "SERV-INFO-001",
                "unidad_medida": "horas",
            }
        ],
        "receptor": {
            "nombre": "Empresa Ejemplo SL",
            "cif": "B12345678",
            "actividad_cnae": "6201 - Programación informática",
            "sector": "Tecnología",
            "tipo_empresa": "SL",
        },
        "emisor": {
            "nombre": "TechConsult SL",
            "cif": "A98765432",
            "actividad_cnae": "6202 - Consultoría informática",
            "pais_residencia": "España",
        },
        "importes": {
            "base_imponible": 1500,
            "iva_21": 315,
            "irpf": 225,
            "total_factura": 1815,
        },
        "contexto_empresarial": {
            "departamento": "IT",
            "centro_coste": "CC-001",
            "proyecto": "PROJ-ERP-2024",
            "porcentaje_afectacion": 100,
            "uso_empresarial": "Exclusivo",
            "justificacion_gasto": "Implementación sistema ERP para mejorar procesos",
        },
        "fiscal": {
            "regimen_iva": "General",
            "retencion_aplicada": True,
            "tipo_retencion": 15,
            "operacion_intracomunitaria": False,
        },
        "relacion_comercial": {
            "tipo_relacion": "Tercero independiente",
            "empresa_vinculada": False,
            "operacion_vinculada": False,
            "proveedor_habitual": True,
        },
        "fechas": {
            "emision": "2024-10-15",
            "prestacion_servicio": "2024-10-01",
            "pago_efectivo": "2024-10-20",
        },
    }

    url = "http://212.69.86.224:8001/fiscal/classify"

    # Serialize payload ensuring UTF-8 characters are preserved as-is
    data_json = json.dumps(payload, ensure_ascii=False)

    cmd = [
        "curl",
        "-sS",  # silent but show errors
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "-d",
        data_json,
    ]

    try:
        # Stream output directly to the terminal
        proc = subprocess.run(cmd)
        return proc.returncode
    except FileNotFoundError:
        sys.stderr.write(
            "Error: 'curl' is not installed or not found in PATH.\n"
        )
        return 127


if __name__ == "__main__":
    sys.exit(main())

