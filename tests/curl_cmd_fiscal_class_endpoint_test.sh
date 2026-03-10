curl -X POST http://212.69.86.224:8001/fiscal/classify \
  -H "Content-Type: application/json" \
  -d '{
  "identificacion": {
    "id_interno": "FACT-2024-001234",
    "numero_factura": "F-2024-0156",
    "serie": "A"
  },
  "conceptos": [
    {
      "descripcion": "Servicios consultoría informática desarrollo ERP",
      "cantidad": 40,
      "precio_unitario": 37.5,
      "importe_linea": 1500,
      "tipo_iva": 21,
      "codigo_producto": "SERV-INFO-001",
      "unidad_medida": "horas"
    }
  ],
  "receptor": {
    "nombre": "Empresa Ejemplo SL",
    "cif": "B12345678",
    "actividad_cnae": "6201 - Programación informática",
    "sector": "Tecnología",
    "tipo_empresa": "SL"
  },
  "emisor": {
    "nombre": "TechConsult SL",
    "cif": "A98765432",
    "actividad_cnae": "6202 - Consultoría informática",
    "pais_residencia": "España"
  },
  "importes": {
    "base_imponible": 1500,
    "iva_21": 315,
    "irpf": 225,
    "total_factura": 1815
  },
  "contexto_empresarial": {
    "departamento": "IT",
    "centro_coste": "CC-001",
    "proyecto": "PROJ-ERP-2024",
    "porcentaje_afectacion": 100,
    "uso_empresarial": "Exclusivo",
    "justificacion_gasto": "Implementación sistema ERP para mejorar procesos"
  },
  "fiscal": {
    "regimen_iva": "General",
    "retencion_aplicada": true,
    "tipo_retencion": 15,
    "operacion_intracomunitaria": false
  },
  "relacion_comercial": {
    "tipo_relacion": "Tercero independiente",
    "empresa_vinculada": false,
    "operacion_vinculada": false,
    "proveedor_habitual": true
  },
  "fechas": {
    "emision": "2024-10-15",
    "prestacion_servicio": "2024-10-01",
    "pago_efectivo": "2024-10-20"
  }
}' | jq