#!/usr/bin/env python3
"""
Lanza N peticiones a fiscal/classify en paralelo, con un máximo de C
simultáneamente, y muestra un resumen estadístico.

Uso:
  python tests/run_fiscal_classify_parallel.py              # 1000 req, 20 hilos
  python tests/run_fiscal_classify_parallel.py -n 2000      # 2000 req
  python tests/run_fiscal_classify_parallel.py -j 10        # 10 concurrencia
  python tests/run_fiscal_classify_parallel.py --print      # imprime cada respuesta
  python tests/run_fiscal_classify_parallel.py --unique     # variación única por petición (id/num factura)
  python tests/run_fiscal_classify_parallel.py --print-errors  # imprime solo body en errores
  python tests/run_fiscal_classify_parallel.py --retries 3 --connect-timeout 5 --timeout 30  # más robusto
  python tests/run_fiscal_classify_parallel.py --debug      # imprime diagnóstico por petición en errores

Notas:
  - Requiere 'curl' en PATH.
  - Los tiempos se calculan con -w de curl (time_total).
  - Las latencias se resumen solo para respuestas con HTTP 200.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple


CURL_EXIT_REASONS = {
    0: "OK",
    6: "Could not resolve host (DNS)",
    7: "Failed to connect (refused/unreachable)",
    28: "Operation timed out",
    35: "SSL connect error",
    52: "Empty reply from server",
    56: "Failure in receiving network data",
    60: "Peer's certificate cannot be authenticated",
}


def build_payload(index: int | None = None) -> dict:
    suffix = f"-{index:04d}" if index is not None else ""
    return {
        "identificacion": {
            "id_interno": f"FACT-2024-001234{suffix}",
            "numero_factura": f"F-2024-0156{suffix}",
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


def run_once(
    url: str,
    data_json: str,
    max_time: int = 60,
    print_body: bool = False,
    connect_timeout: int | None = None,
    retries: int = 0,
    retry_all_errors: bool = False,
    verbose: bool = False,
) -> Tuple[int, float, str, str, int]:
    """Ejecuta una petición y retorna (status_code, time_total, stdout, stderr, exit_code)."""
    format_str = "\n__CURL_STATUS:%{http_code}__\n__CURL_TIME:%{time_total}__\n"
    cmd = [
        "curl",
        "-sS",
        "-m",
        str(max_time),
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
    if connect_timeout and connect_timeout > 0:
        cmd.extend(["--connect-timeout", str(connect_timeout)])
    if retries and retries > 0:
        cmd.extend(["--retry", str(retries)])
        if retry_all_errors:
            cmd.append("--retry-all-errors")
    if verbose:
        cmd.append("-v")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout or ""
    err = proc.stderr or ""
    status_code = 0
    time_total = 0.0
    # body defaults to stdout; stderr is returned separately for diagnóstico

    try:
        if "__CURL_STATUS:" in out and "__CURL_TIME:" in out:
            before_status, _, rest = out.rpartition("__CURL_STATUS:")
            status_part, _, time_part = rest.partition("__CURL_TIME:")
            out = before_status.rstrip("\n")
            status_code_str = status_part.split("__", 1)[0].strip()
            time_str = time_part.split("__", 1)[0].strip()
            status_code = int(status_code_str or 0)
            time_total = float(time_str or 0.0)
        else:
            status_code = 0
            time_total = 0.0
    except Exception:
        status_code = 0
        time_total = 0.0

    if not print_body:
        out = ""
        err = ""
    return status_code, time_total, out, err, proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark paralelo fiscal/classify")
    parser.add_argument("-n", type=int, default=1000, help="Número total de peticiones (default: 1000)")
    parser.add_argument("-j", "--concurrency", type=int, default=20, help="Concurrencia máx. (default: 20)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout por petición en segundos (default: 60)")
    parser.add_argument("--connect-timeout", type=int, default=5, help="Timeout de conexión TCP (default: 5)")
    parser.add_argument("--retries", type=int, default=0, help="Reintentos de curl por error transitorio (default: 0)")
    parser.add_argument("--retry-all-errors", action="store_true", help="Reintentar en todos los errores (curl --retry-all-errors)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Modo verbose de curl (-v)")
    parser.add_argument("--print", dest="print_body", action="store_true", help="Imprime el body de cada respuesta")
    parser.add_argument("--print-errors", dest="print_errors", action="store_true", help="Imprime solo el body en respuestas no-200")
    parser.add_argument("--unique", action="store_true", help="Usa payloads únicos por petición (evita duplicados)")
    args = parser.parse_args()

    if args.concurrency < 1:
        sys.stderr.write("Concurrencia debe ser >= 1\n")
        return 2

    payload = build_payload()
    data_json = json.dumps(payload, ensure_ascii=False)
    url = "http://212.69.86.224:8001/fiscal/classify"

    status_counts = Counter()
    exit_code_counts = Counter()
    times: list[float] = []
    successes = 0

    print(f"Lanzando {args.n} peticiones a {url} con concurrencia={args.concurrency} …")

    futures = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for i in range(args.n):
                if args.unique:
                    dj = json.dumps(build_payload(i + 1), ensure_ascii=False)
                else:
                    dj = data_json
                futures.append(
                    ex.submit(
                        run_once,
                        url,
                        dj,
                        args.timeout,
                        args.print_body or args.print_errors,
                        args.connect_timeout,
                        args.retries,
                        args.retry_all_errors,
                        args.verbose,
                    )
                )

            completed = 0
            for fut in as_completed(futures):
                completed += 1
                try:
                    status, t, out, err, exit_code = fut.result()
                except FileNotFoundError:
                    sys.stderr.write("Error: 'curl' no está instalado o no está en PATH.\n")
                    return 127
                except Exception as e:
                    # Cuenta como fallo genérico
                    status = 0
                    t = 0.0
                    out = ""
                    err = str(e)
                    exit_code = 1

                exit_code_counts[exit_code] += 1
                status_counts[status] += 1
                if status == 200:
                    successes += 1
                    times.append(t)

                if (args.print_body or (args.print_errors and status != 200)) and (out or err):
                    to_print = out if out else err
                    print(to_print)

                if status != 200 and (args.verbose or args.print_errors or '--debug' in sys.argv):
                    reason = CURL_EXIT_REASONS.get(exit_code, "Unknown error")
                    time_ms = int(t * 1000) if t else 0
                    print(f"[diag] status={status} exit={exit_code} ({reason}) time={time_ms}ms")

                if completed % 50 == 0 or completed == args.n:
                    print(f"Progreso: {completed}/{args.n} (200 OK: {successes})")
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")

    total = args.n
    errors = total - successes

    print("\nResumen:")
    print(f"  Total:\t{total}")
    print(f"  200 OK:\t{successes}")
    print(f"  Errores:\t{errors}")

    print("  Por código de estado:")
    for code, cnt in sorted(status_counts.items()):
        label = "(OK)" if code == 200 else ""
        print(f"    {code} {label}\t{cnt}")

    print("  Por exit code de curl:")
    for code, cnt in sorted(exit_code_counts.items()):
        print(f"    {code}\t{cnt}")

    if times:
        avg = statistics.mean(times)
        p50 = statistics.median(times)
        sorted_times = sorted(times)
        def pct(p: float) -> float:
            idx = max(0, min(len(sorted_times) - 1, int(p * len(sorted_times)) - 1))
            return sorted_times[idx]
        p90 = pct(0.90)
        p95 = pct(0.95)
        print("  Latencias (solo éxitos):")
        print(f"    Promedio:\t{int(avg * 1000)} ms")
        print(f"    Mediana:\t{int(p50 * 1000)} ms")
        print(f"    P90:\t{int(p90 * 1000)} ms")
        print(f"    P95:\t{int(p95 * 1000)} ms")
        print(f"    Min:\t{int(min(sorted_times) * 1000)} ms")
        print(f"    Max:\t{int(max(sorted_times) * 1000)} ms")

    return 0 if successes == total else 1


if __name__ == "__main__":
    sys.exit(main())
