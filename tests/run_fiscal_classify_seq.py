#!/usr/bin/env python3
"""
Ejecuta la petición fiscal/classify 20 veces (por defecto) de forma secuencial
y muestra un resumen estadístico (cuántos 200, cuántos errores por código, tiempos).

Uso:
  python tests/run_fiscal_classify_seq.py            # 20 repeticiones
  python tests/run_fiscal_classify_seq.py -n 50      # 50 repeticiones
  python tests/run_fiscal_classify_seq.py --print    # imprime el body de cada respuesta
"""

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter


def build_payload() -> dict:
    return {
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


def run_once(url: str, data_json: str) -> tuple[int, float, str, int]:
    """Ejecuta una vez y retorna (status_code, tiempo_total, body, exit_code)."""
    # Inserta marcadores para extraer http_code y time_total del output
    format_str = "\n__CURL_STATUS:%{http_code}__\n__CURL_TIME:%{time_total}__\n"
    cmd = [
        "curl",
        "-sS",  # silencioso pero muestra errores
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
        "-d",
        data_json,
        "-w",
        format_str,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout
    status_code = 0
    time_total = 0.0

    body = out
    # Extrae marcadores si están presentes
    try:
        # Separar por el último marcador de status para tolerar posibles apariciones en el body
        if "__CURL_STATUS:" in out and "__CURL_TIME:" in out:
            before_status, _, rest = out.rpartition("__CURL_STATUS:")
            status_part, _, time_part = rest.partition("__CURL_TIME:")
            # Limpia body (todo lo anterior a los marcadores menos saltos extra)
            body = before_status.rstrip("\n")
            # Parseos
            status_code_str = status_part.split("__", 1)[0].strip()
            time_str = time_part.split("__", 1)[0].strip()
            status_code = int(status_code_str or 0)
            time_total = float(time_str or 0.0)
        else:
            # Sin marcadores: curl pudo fallar antes de emitir -w
            status_code = 0
            time_total = 0.0
    except Exception:
        status_code = 0
        time_total = 0.0

    return status_code, time_total, body, proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark secuencial fiscal/classify")
    parser.add_argument("-n", type=int, default=20, help="Número de repeticiones (default: 20)")
    parser.add_argument("--print", dest="print_body", action="store_true", help="Imprime el body de cada respuesta")
    args = parser.parse_args()

    payload = build_payload()
    data_json = json.dumps(payload, ensure_ascii=False)
    url = "http://212.69.86.224:8001/fiscal/classify"

    status_counts = Counter()
    exit_code_counts = Counter()
    times: list[float] = []
    successes = 0

    print(f"Ejecutando {args.n} peticiones secuenciales a {url}…")

    for i in range(1, args.n + 1):
        try:
            status, t, body, exit_code = run_once(url, data_json)
        except FileNotFoundError:
            sys.stderr.write("Error: 'curl' no está instalado o no está en PATH.\n")
            return 127

        exit_code_counts[exit_code] += 1
        status_counts[status] += 1
        if status == 200:
            successes += 1
            times.append(t)

        time_ms = int(t * 1000) if t else 0
        print(f"[{i:02d}] status={status} tiempo={time_ms} ms exit={exit_code}")
        if args.print_body:
            print(body)

    total = args.n
    errors = total - successes

    print("\nResumen:")
    print(f"  Total:	{total}")
    print(f"  200 OK:	{successes}")
    print(f"  Errores:	{errors}")

    # Desglose por status
    print("  Por código de estado:")
    for code, cnt in sorted(status_counts.items()):
        label = "(OK)" if code == 200 else ""
        print(f"    {code} {label}\t{cnt}")

    # Desglose por exit code de curl
    print("  Por exit code de curl:")
    for code, cnt in sorted(exit_code_counts.items()):
        print(f"    {code}\t{cnt}")

    if times:
        avg = statistics.mean(times)
        p50 = statistics.median(times)
        p90 = sorted(times)[max(0, int(0.9 * len(times)) - 1)]
        print("  Latencias (solo éxitos):")
        print(f"    Promedio:\t{int(avg * 1000)} ms")
        print(f"    Mediana:\t{int(p50 * 1000)} ms")
        print(f"    P90:\t{int(p90 * 1000)} ms")

    # Código de salida: 0 si hubo al menos un 200 y no hubo errores de ejecución graves
    return 0 if successes == total else 1


if __name__ == "__main__":
    sys.exit(main())

