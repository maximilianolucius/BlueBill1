#!/usr/bin/env python3
"""Script utilitario para lanzar 4 facturas en batch contra /fiscal/classify_async.

Ejemplo de uso:
    python tests/run_classify_async_batch.py

La URL base se controla con BLUEBILL_API_URL (por defecto http://localhost:8001).
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any, List, Tuple

import httpx


BASE_URL = os.getenv("BLUEBILL_API_URL", "http://localhost:8001").rstrip("/")
POLL_INTERVAL_SECONDS = float(os.getenv("BLUEBILL_POLL_INTERVAL", "5"))
POLL_TIMEOUT_SECONDS = int(os.getenv("BLUEBILL_POLL_TIMEOUT", "600"))
BATCH_SIZE = int(os.getenv("BLUEBILL_BATCH_SIZE", "5"))


def build_sample_factura(index: int) -> Dict[str, Any]:
    """Genera una factura de prueba variando algunos campos para cada solicitud."""
    return {
        "identificacion": {
            "numero_factura": f"TEST-BATCH-{index:03d}",
            "serie": "TB",
        },
        "conceptos": [
            {
                "descripcion": f"Servicio de consultoría tecnológica #{index}",
                "cantidad": 1,
                "precio_unitario": 1200.0 + index * 50,
                "importe_linea": 1200.0 + index * 50,
                "codigo_producto": f"CONSULT-{index:02d}",
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
            "base_imponible": 1200.0 + index * 50,
            "iva_21": round((1200.0 + index * 50) * 0.21, 2),
            "irpf": 0.0,
            "total_factura": round((1200.0 + index * 50) * 1.21, 2),
        },
        "contexto_empresarial": {
            "departamento": "IT",
            "centro_coste": f"CC-IT-{index:03d}",
            "proyecto": "Digitalización",
            "porcentaje_afectacion": 100,
            "uso_empresarial": "Exclusivo",
            "justificacion_gasto": "Servicios profesionales para proyecto de transformación digital.",
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
            "fecha_emision": "2024-01-20",
            "fecha_vencimiento": "2024-02-19",
        },
    }


def pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


async def submit_job(client: httpx.AsyncClient, index: int) -> Tuple[str, Dict[str, Any]]:
    """Envía una factura y devuelve el job_id junto con la respuesta inicial."""
    payload = build_sample_factura(index)
    response = await client.post(f"{BASE_URL}/fiscal/classify_async", json=payload)
    response.raise_for_status()
    job_info = response.json()
    job_id = job_info["job_id"]
    print(f"[{job_id}] ✅ Lanzado para factura #{index}")
    return job_id, job_info


async def poll_job(client: httpx.AsyncClient, job_id: str) -> Dict[str, Any]:
    """Consulta el estado y el resultado de un job hasta que finalice."""
    deadline = asyncio.get_event_loop().time() + POLL_TIMEOUT_SECONDS

    while asyncio.get_event_loop().time() < deadline:
        status_resp = await client.get(f"{BASE_URL}/fiscal/status/{job_id}")
        status_resp.raise_for_status()
        status_payload = status_resp.json()

        print(
            f"[{job_id}] Estado: {status_payload['status']}, "
            f"Progreso: {status_payload['progress']}%, "
            f"Paso: {status_payload['current_step']}"
        )

        if status_payload["status"] in {"completed", "failed"}:
            result_resp = await client.get(f"{BASE_URL}/fiscal/result/{job_id}")
            result_resp.raise_for_status()
            return result_resp.json()

        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(f"El job {job_id} no finalizó tras {POLL_TIMEOUT_SECONDS}s")


async def run_batch(batch_size: int) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Lanza un batch y espera a que todos los jobs finalicen."""
    async with httpx.AsyncClient(timeout=40) as client:
        submissions = [
            await submit_job(client, idx + 1)
            for idx in range(batch_size)
        ]

        poll_tasks = [
            poll_job(client, job_id)
            for job_id, _ in submissions
        ]

        results = await asyncio.gather(*poll_tasks, return_exceptions=True)

    aggregated: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for (job_id, job_info), result in zip(submissions, results):
        aggregated.append((job_id, job_info, result))
    return aggregated


async def async_main() -> int:
    print(f"Usando servicio en: {BASE_URL}")
    print(f"Lanzando batch de {BATCH_SIZE} facturas...\n")

    for _ in range(4):
        aggregated = await run_batch(BATCH_SIZE)

        exit_code = 0
        for job_id, launch_payload, result in aggregated:
            print("=" * 80)
            print(f"Job {job_id}")
            print("Solicitud inicial:")
            print(pretty_json(launch_payload))
            if isinstance(result, Exception):
                exit_code = 1
                print("Resultado: ❌ Error")
                print(repr(result))
            else:
                status = result.get("status")
                print(f"Resultado ({status}):")
                print(pretty_json(result))
        if exit_code: break
    return exit_code


def main() -> int:
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
