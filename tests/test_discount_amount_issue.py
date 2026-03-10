#!/usr/bin/env python
"""
Clasifica todas las facturas incluidas en facturas_transformadas.json usando el endpoint
/fiscal/classify expuesto por bluebill_app.py.

Uso:
    python tests/test_discount_amount_issue.py

Requiere que el servicio FastAPI esté en ejecución. Puede configurarse la base
del endpoint mediante la variable de entorno BLUEBILL_BASE_URL.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8001"
REQUEST_TIMEOUT = float(os.getenv("BLUEBILL_REQUEST_TIMEOUT", "360"))
POLL_INTERVAL = float(os.getenv("BLUEBILL_POLL_INTERVAL", "2"))
MAX_POLL_TIME = float(os.getenv("BLUEBILL_MAX_POLL_TIME", "420"))


@dataclass
class ClassificationResult:
    factura_id: str
    numero_factura: str
    descuento: float
    codigo_principal: Optional[str]
    confianza: Optional[float]
    status: str
    detail: Optional[str] = None


def load_invoices(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if "facturas" in payload and isinstance(payload["facturas"], list):
            return payload["facturas"]

    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, dict) and entry.get("type") == "table" and entry.get("name") == "fg":
                return entry.get("data", [])

    raise RuntimeError(f"No se encontraron facturas en {json_path}")


def parse_number(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        value = value.replace(",", ".")
        return float(value)
    except Exception:
        return None


def parse_json_field(raw: Any) -> Optional[Any]:
    if not isinstance(raw, str):
        return raw
    raw = raw.strip()
    if not raw or raw.lower() in {"null", "none"}:
        return None
    if raw[0] not in "[{":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def clean_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if v not in (None, "", [], {}, ())}


def build_factura_payload(invoice: Dict[str, Any]) -> Dict[str, Any]:
    conceptos_raw = invoice.get("conceptos") or []
    concepto_data = conceptos_raw[0] if conceptos_raw and isinstance(conceptos_raw[0], dict) else {}

    concepto_desc = (
        concepto_data.get("descripcion")
        or invoice.get("item_descripcion")
        or invoice.get("nombre_item")
        or "Concepto sin descripción"
    )
    cantidad = parse_number(concepto_data.get("cantidad")) or parse_number(invoice.get("cantidad")) or 1.0

    precio_unitario = (
        parse_number(concepto_data.get("precio_unitario"))
        or parse_number(invoice.get("precio_unitario"))
    )
    if precio_unitario is None:
        precio_unitario = parse_number(concepto_data.get("importe")) or parse_number(invoice.get("base_imponible")) or 0.0

    importe_total_linea = (
        parse_number(concepto_data.get("importe_linea"))
        or parse_number(concepto_data.get("importe"))
        or parse_number(invoice.get("importe_total_linea"))
    )
    if importe_total_linea is None:
        importe_total_linea = round(cantidad * precio_unitario, 2)

    tipo_iva = parse_number(invoice.get("tipo_iva"))
    if tipo_iva is None:
        tipo_iva = parse_number(concepto_data.get("tipo_iva"))
    if tipo_iva is None:
        tipo_iva = 21

    concepto = clean_dict(
        {
            "descripcion": concepto_desc,
            "cantidad": cantidad,
            "precio_unitario": precio_unitario,
            "importe_linea": importe_total_linea,
            "descuento": parse_number(concepto_data.get("descuento")) or parse_number(invoice.get("descuento")),
            "impuesto": parse_number(concepto_data.get("impuesto")) or parse_number(invoice.get("impuesto"))
            or parse_number((invoice.get("importes") or {}).get("iva")),
            "tipo_iva": tipo_iva,
        }
    )

    identificacion_data = invoice.get("identificacion") or {}
    identificacion = clean_dict(
        {
            "factura_id": invoice.get("factura_id") or identificacion_data.get("factura_id"),
            "numero_factura": invoice.get("numero_factura") or identificacion_data.get("numero_factura"),
            "fecha_emision": invoice.get("fecha_emision")
            or identificacion_data.get("fecha_emision")
            or (invoice.get("fechas") or {}).get("emision"),
            "factura_ocr_id": invoice.get("factura_ocr_id") or identificacion_data.get("factura_ocr_id"),
            "job_id": invoice.get("job_id"),
        }
    )

    receptor_data = invoice.get("receptor") or {}
    receptor = clean_dict(
        {
            "nombre": receptor_data.get("nombre") or invoice.get("nombre_receptor"),
            "cif": receptor_data.get("cif") or invoice.get("cif_receptor"),
            "actividad_cnae": receptor_data.get("actividad_cnae"),
            "direccion": receptor_data.get("direccion") or invoice.get("direccion_receptor"),
            "email": receptor_data.get("email") or invoice.get("email_receptor"),
            "sector": receptor_data.get("sector"),
            "tipo_empresa": receptor_data.get("tipo_empresa"),
        }
    )

    emisor_data = invoice.get("emisor") or {}
    emisor = clean_dict({
        "nombre": emisor_data.get("nombre") or invoice.get("nombre_emisor"),
        "cif": emisor_data.get("cif") or invoice.get("cif_emisor"),
        "actividad_cnae": emisor_data.get("actividad_cnae"),
        "direccion": emisor_data.get("direccion") or invoice.get("direccion_emisor"),
        "email": emisor_data.get("email") or invoice.get("email_emisor"),
    })

    importes_data = invoice.get("importes") or {}
    importes = clean_dict(
        {
            "base_imponible": parse_number(importes_data.get("base_imponible"))
            or parse_number(invoice.get("base_imponible")),
            "total_impuestos": parse_number(importes_data.get("iva"))
            or parse_number(invoice.get("total_impuestos")),
            "total_irpf": parse_number(importes_data.get("irpf")) or parse_number(invoice.get("total_irpf")),
            "total_factura": parse_number(importes_data.get("total"))
            or parse_number(importes_data.get("total_factura"))
            or parse_number(invoice.get("total_factura"))
            or importe_total_linea,
        }
    )

    metadata = parse_json_field(invoice.get("metadata_completo")) or {}
    porcentaje = metadata.get("afectacion_empresarial")
    if isinstance(porcentaje, str) and porcentaje.endswith("%"):
        porcentaje = porcentaje.strip("%")
    porcentaje_afectacion = parse_number(porcentaje) or 100.0

    contexto_data = invoice.get("contexto_empresarial") or {}
    contexto_empresarial = clean_dict(
        {
            "departamento": metadata.get("departamento_origen") or contexto_data.get("departamento"),
            "proyecto": metadata.get("proyecto_asociado") or contexto_data.get("proyecto"),
            "porcentaje_afectacion": porcentaje_afectacion,
            "uso_empresarial": metadata.get("uso_empresarial")
            or contexto_data.get("uso_empresarial")
            or "Exclusivo",
            "justificacion_gasto": metadata.get("justificacion_gasto") or contexto_data.get("justificacion_gasto"),
        }
    )

    fiscal_data = invoice.get("fiscal") or {}
    fiscal = clean_dict(
        {
            "regimen_iva": fiscal_data.get("regimen_iva") or "General",
            "retencion_aplicada": bool(
                parse_number(fiscal_data.get("tipo_retencion"))
                or parse_number(invoice.get("total_irpf"))
            ),
            "tipo_retencion": parse_number(fiscal_data.get("tipo_retencion")) or 0,
            "operacion_intracomunitaria": bool(fiscal_data.get("operacion_intracomunitaria")),
            "exencion_aplicada": bool(fiscal_data.get("exencion_aplicada")),
        }
    )

    fechas_data = invoice.get("fechas") or {}
    fechas = clean_dict(
        {
            "emision": fechas_data.get("emision") or invoice.get("fecha_emision"),
            "prestacion_servicio": fechas_data.get("prestacion_servicio"),
            "pago_efectivo": fechas_data.get("pago_efectivo"),
            "registro": invoice.get("factura_created_at"),
            "actualizacion": invoice.get("factura_updated_at"),
        }
    )

    relacion_data = invoice.get("relacion_comercial") or {}
    relacion = clean_dict(
        {
            "tipo_relacion": relacion_data.get("tipo_relacion") or "Tercero independiente",
            "empresa_vinculada": bool(relacion_data.get("empresa_vinculada")),
            "operacion_vinculada": bool(relacion_data.get("operacion_vinculada")),
            "proveedor_habitual": bool(relacion_data.get("proveedor_habitual")),
        }
    )

    payload = clean_dict(
        {
            "identificacion": identificacion,
            "conceptos": [concepto],
            "receptor": receptor or None,
            "emisor": emisor or None,
            "importes": importes or None,
            "contexto_empresarial": contexto_empresarial or None,
            "fiscal": fiscal or None,
            "relacion_comercial": relacion or None,
            "fechas": fechas or None,
        }
    )

    return payload


def classify_invoice(session: requests.Session, base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    classify_url = f"{base_url}/fiscal/classify"
    result_url_template = f"{base_url}/fiscal/result/{{job_id}}"

    response = session.post(classify_url, json=payload, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    data = response.json()
    if "clasificacion" in data:
        return data

    job_id = data.get("job_id")
    if not job_id:
        raise RuntimeError(f"Respuesta inesperada: {data}")

    deadline = time.monotonic() + MAX_POLL_TIME
    result_url = result_url_template.format(job_id=job_id)

    while time.monotonic() < deadline:
        job_resp = session.get(result_url, timeout=REQUEST_TIMEOUT)
        if job_resp.status_code == 200:
            job_data = job_resp.json()
            status = job_data.get("status")
            if status == "completed" and job_data.get("result"):
                return job_data["result"]
            if status == "failed":
                detail = job_data.get("error") or "Job failed sin detalle"
                raise RuntimeError(detail)
        elif job_resp.status_code != 202:
            raise RuntimeError(f"Error consultando job: HTTP {job_resp.status_code} {job_resp.text}")

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"Timeout esperando resultado del job {job_id}")


def iter_invoices(invoices: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for invoice in invoices:
        if isinstance(invoice, dict):
            yield invoice


def main() -> int:
    base_url = os.getenv("BLUEBILL_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    json_path = Path(__file__).resolve().parents[1] / "data" / "facturas" / "facturas_transformadas.json"

    try:
        invoices = list(iter_invoices(load_invoices(json_path)))
    except Exception as exc:
        print(f"[ERROR] No se pudieron cargar las facturas: {exc}", file=sys.stderr)
        return 1

    total = len(invoices)
    if total == 0:
        print("No se encontraron facturas en facturas_transformadas.json")
        return 1

    print(f"Clasificando {total} facturas contra {base_url}/fiscal/classify ...")
    session = requests.Session()
    results: List[ClassificationResult] = []

    for idx, invoice in enumerate(invoices, start=1):
        payload = build_factura_payload(invoice)
        factura_id = (
            invoice.get("factura_id")
            or (invoice.get("identificacion") or {}).get("factura_id")
            or invoice.get("numero_factura")
            or (invoice.get("identificacion") or {}).get("numero_factura")
            or f"factura_{idx:04d}"
        )
        numero_factura = (
            invoice.get("numero_factura")
            or (invoice.get("identificacion") or {}).get("numero_factura")
            or f"N/A_{idx:04d}"
        )
        descuento = (
            parse_number(invoice.get("descuento"))
            or parse_number((invoice.get("conceptos") or [{}])[0].get("descuento"))
            or 0.0
        )

        try:
            classification = classify_invoice(session, base_url, payload)
            clasif_info = classification.get("clasificacion", {})
            codigo = clasif_info.get("codigo_principal")
            confianza = clasif_info.get("confianza")

            # Extract incentivo_idi information
            oportunidades = classification.get("oportunidades_fiscales", [])
            incentivo_info = ""
            for oportunidad in oportunidades:
                if oportunidad.get("tipo") == "incentivo_idi":
                    ahorro = oportunidad.get("ahorro_estimado_euros", 0)
                    aplicabilidad = oportunidad.get("aplicabilidad", "N/A")
                    incentivo_info = f" I+D+i: €{ahorro} ({aplicabilidad})"
                    break

            results.append(
                ClassificationResult(
                    factura_id=str(factura_id),
                    numero_factura=str(numero_factura),
                    descuento=descuento,
                    codigo_principal=codigo,
                    confianza=confianza,
                    status="ok",
                )
            )

            conf_text = f"{confianza:.2f}" if isinstance(confianza, (int, float)) else "N/D"
            print(f"[{idx}/{total}] Factura {factura_id} descuento={descuento:.2f} -> {codigo or 'SIN CODIGO'} (conf={conf_text}){incentivo_info}")

        except Exception as exc:
            detail = str(exc)
            results.append(
                ClassificationResult(
                    factura_id=str(factura_id),
                    numero_factura=str(numero_factura),
                    descuento=descuento,
                    codigo_principal=None,
                    confianza=None,
                    status="error",
                    detail=detail,
                )
            )
            print(f"[{idx}/{total}] Factura {factura_id} ERROR: {detail}", file=sys.stderr)

        if idx > 50:
            break

    ok_count = sum(1 for r in results if r.status == "ok")
    error_count = total - ok_count
    print(f"\nFinalizado. Exitos: {ok_count}  Errores: {error_count}")

    if error_count:
        print("Facturas con error:")
        for item in results:
            if item.status != "error":
                continue
            print(f"  - {item.factura_id} ({item.numero_factura}): {item.detail}")

    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
