#!/usr/bin/env python3
"""Script utilitario para lanzar una sola factura contra /fiscal/classify_async.

Ejemplo de uso:
    python tests/run_classify_async_single.py

Puedes ajustar la URL del servicio exponiendo la variable de entorno
BLUEBILL_API_URL (por defecto http://localhost:8000).
"""

import os
import sys
import json
import time
from typing import Dict, Any
from datetime import datetime

import httpx


BASE_URL = os.getenv("BLUEBILL_API_URL", "http://localhost:8001").rstrip("/")
POLL_INTERVAL_SECONDS = float(os.getenv("BLUEBILL_POLL_INTERVAL", "5"))
POLL_TIMEOUT_SECONDS = int(os.getenv("BLUEBILL_POLL_TIMEOUT", "600"))


def build_sample_factura() -> Dict[str, Any]:
    """Genera una factura de prueba con los campos mínimos requeridos."""
    return {
        "identificacion": {
            "numero_factura": "TEST-ASYNC-001",
            "serie": "TST",
        },
        "conceptos": [
            {
                "descripcion": "Servicios de consultoría tecnológica",
                "cantidad": 1,
                "precio_unitario": 1500.0,
                "importe_linea": 1500.0,
                "codigo_producto": "CONSULT-IT",
                "unidad_medida": "unidad",
            }
        ],
        "receptor": {
            "nombre": "Empresa Receptora SL",
            "cif": "B12345678",
            "actividad_cnae": "6201",
            "sector": "Tecnología",
            "tipo_empresa": "Sociedad Limitada",
            "pais_residencia": "España",
        },
        "emisor": {
            "nombre": "Proveedor Servicios Digitales SA",
            "cif": "A87654321",
            "actividad_cnae": "6202",
            "sector": "Consultoría",
            "tipo_empresa": "Sociedad Anónima",
            "pais_residencia": "España",
        },
        "importes": {
            "base_imponible": 1500.0,
            "iva_21": 315.0,
            "irpf": 0.0,
            "total_factura": 1815.0,
        },
        "contexto_empresarial": {
            "departamento": "IT",
            "centro_coste": "CC-IT-001",
            "proyecto": "Implementación ERP",
            "porcentaje_afectacion": 100,
            "uso_empresarial": "Exclusivo",
            "justificacion_gasto": "Implementación de un nuevo ERP corporativo.",
        },
        "fiscal": {
            "regimen_iva": "General",
            "retencion_aplicada": False,
            "tipo_retencion": 0,
            "operacion_intracomunitaria": False,
        },
        "relacion_comercial": {
            "operacion_vinculada": False,
        },
        "fechas": {
            "fecha_emision": "2024-01-15",
            "fecha_vencimiento": "2024-02-14",
        },
    }


def pretty_json(data: Dict[str, Any]) -> str:
    """Formatea un dict como JSON indentado."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def poll_job(client: httpx.Client, job_id: str) -> Dict[str, Any]:
    """Consulta /fiscal/status y /fiscal/result hasta que finalice o agote el timeout."""
    deadline = time.time() + POLL_TIMEOUT_SECONDS

    while time.time() < deadline:
        status_resp = client.get(f"{BASE_URL}/fiscal/status/{job_id}", timeout=30)
        status_resp.raise_for_status()
        status_payload = status_resp.json()

        print(f"[{datetime.utcnow().isoformat()}Z] "
              f"[{job_id}] Estado: {status_payload['status']}, "
              f"Progreso: {status_payload['progress']}%, "
              f"Paso: {status_payload['current_step']}")

        if status_payload["status"] in {"completed", "failed"}:
            result_resp = client.get(f"{BASE_URL}/fiscal/result/{job_id}", timeout=30)
            result_resp.raise_for_status()
            return result_resp.json()

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"El job {job_id} no finalizó tras {POLL_TIMEOUT_SECONDS}s")


def main() -> int:
    print(f"Usando servicio en: {BASE_URL}")

    payload = build_sample_factura()
    overall_start = time.perf_counter()

    with httpx.Client(timeout=40) as client:
        response = client.post(f"{BASE_URL}/fiscal/classify_async", json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            print(f"❌ Error al lanzar job: {exc.response.text}", file=sys.stderr)
            return 1

        job_info = response.json()
        job_id = job_info["job_id"]

        print("✅ Job lanzado correctamente")
        print(pretty_json(job_info))

        try:
            result_payload = poll_job(client, job_id)
        except TimeoutError as exc:
            print(f"⚠️ {exc}", file=sys.stderr)
            print(f"⏱️ Tiempo transcurrido: {time.perf_counter() - overall_start:.2f}s")
            return 2

    print("📦 Respuesta final del job:")
    print(pretty_json(result_payload))
    print(f"⏱️ Tiempo total de ejecución: {time.perf_counter() - overall_start:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
