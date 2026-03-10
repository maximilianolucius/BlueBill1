#!/usr/bin/env python3
"""
Benchmark simple para ejecutar una consulta HTTP (definida en webquery.txt)
30 veces seguidas y medir tiempos: min, max, media, desviación estándar.

Formato soportado de webquery.txt:
- Comando curl completo (una o varias líneas con continuaciones '\\')
  Ej.: curl -X POST 'http://host:8001/fiscal/classify' -H 'Content-Type: application/json' -d '{...}'
- O bien:
  1ª línea: URL (http/https)
  Resto:   Cuerpo JSON (se enviará como POST application/json)
"""

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests  # type: ignore
except Exception:
    requests = None


def load_query_text(path: Path) -> str:
    text = path.read_text(encoding='utf-8').strip()
    return text


def is_curl(text: str) -> bool:
    return text.lstrip().startswith('curl ')


def normalize_curl(text: str) -> str:
    # Unir líneas con continuaciones \
    lines = text.splitlines()
    joined = ' '.join([ln.strip().rstrip('\\') for ln in lines if ln.strip()])
    return joined


def run_curl(cmd: str) -> tuple[float, int, str]:
    # Añadimos flags para medir estado y silenciar cuerpo
    augmented = f"{cmd} -sS -o /dev/null -w HTTP_STATUS:%{{http_code}}"
    t0 = time.perf_counter()
    proc = subprocess.run(augmented, shell=True, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    out = (proc.stdout or '') + (proc.stderr or '')
    status = 0
    marker = 'HTTP_STATUS:'
    if marker in out:
        try:
            status = int(out.split(marker)[-1].strip()[:3])
        except Exception:
            status = 0
    else:
        status = proc.returncode or 0
    return dt, status, out.strip()


def run_url_json(text: str) -> tuple[float, int, str]:
    if requests is None:
        raise RuntimeError("Falta la librería 'requests'. Instala con: pip install requests")
    lines = text.splitlines()
    if not lines:
        raise ValueError('webquery.txt vacío')
    url = lines[0].strip()
    body = '\n'.join(lines[1:]).strip() or '{}'
    headers = {'Content-Type': 'application/json'}
    t0 = time.perf_counter()
    resp = requests.post(url, data=body.encode('utf-8'), headers=headers, timeout=120)
    dt = time.perf_counter() - t0
    return dt, resp.status_code, resp.text[:200]


def main():
    ap = argparse.ArgumentParser(description='Benchmark simple de consultas HTTP a BlueBill API')
    ap.add_argument('--file', default='webquery.txt', help='Ruta a webquery.txt')
    ap.add_argument('--runs', type=int, default=30, help='Número de repeticiones')
    ap.add_argument('--sleep', type=float, default=0.0, help='Pausa entre repeticiones (seg)')
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"No existe {path}", file=sys.stderr)
        sys.exit(1)

    text = load_query_text(path)
    use_curl = is_curl(text)

    if use_curl:
        cmd = normalize_curl(text)
        runner = lambda: run_curl(cmd)
        print(f"Modo: curl\nComando: {cmd}")
    else:
        runner = lambda: run_url_json(text)
        print("Modo: URL + JSON (POST)")

    durations = []
    statuses = []

    for i in range(1, args.runs + 1):
        try:
            dt, status, preview = runner()
            durations.append(dt)
            statuses.append(status)
            print(f"Run {i:02d}: {dt*1000:.1f} ms (status {status})")
        except Exception as e:
            durations.append(float('nan'))
            statuses.append(0)
            print(f"Run {i:02d}: ERROR {e}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    # Filtrar solo tiempos válidos
    valid = [d for d in durations if d == d]
    if valid:
        min_v = min(valid)
        max_v = max(valid)
        mean_v = statistics.fmean(valid)
        std_v = statistics.pstdev(valid) if len(valid) > 1 else 0.0
        print("\nResultados:")
        print(f"  min:  {min_v*1000:.1f} ms")
        print(f"  max:  {max_v*1000:.1f} ms")
        print(f"  mean: {mean_v*1000:.1f} ms")
        print(f"  std:  {std_v*1000:.1f} ms")
    else:
        print("\nSin mediciones válidas")

    # Resumen de estados
    from collections import Counter
    counts = Counter(statuses)
    print("Estados HTTP:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()

