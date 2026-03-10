#!/usr/bin/env python3
"""
Analizador específico para la factura de consultoría informática
Usa los datos del JSON proporcionado para búsquedas optimizadas
"""

import json
from datetime import datetime
from aeat_scraper import AEATConsultaScraper


def analyze_factura_consultoria():
    """Analiza la factura específica de consultoría informática"""

    # Datos de tu factura
    factura_data = {
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
            "direccion": "Calle Mayor 15, 28001 Madrid",
            "actividad_cnae": "6201 - Programación informática",
            "sector": "Tecnología",
            "tipo_empresa": "SL"
        },
        "emisor": {
            "nombre": "TechConsult SL",
            "cif": "A98765432",
            "direccion": "Avenida Tecnología 25, 28002 Madrid",
            "email": "facturacion@techconsult.es",
            "actividad_cnae": "6202 - Consultoría informática",
            "pais_residencia": "España"
        },
        "importes": {
            "base_imponible": 1500,
            "iva_21": 315,
            "irpf": 225,
            "total_factura": 1815
        },
        "fiscal": {
            "regimen_iva": "General",
            "retencion_aplicada": True,
            "tipo_retencion": 15
        },
        "contexto_empresarial": {
            "departamento": "IT",
            "proyecto": "PROJ-ERP-2024",
            "uso_empresarial": "Exclusivo",
            "justificacion_gasto": "Implementación sistema ERP para mejorar procesos"
        }
    }

    # Crear scraper
    scraper = AEATConsultaScraper(headless=False)

    try:
        print("=== ANÁLISIS FISCAL AUTOMATIZADO ===")
        print(f"Factura: {factura_data['identificacion']['numero_factura']}")
        print(f"Concepto: {factura_data['conceptos'][0]['descripcion']}")
        print(f"Importe: {factura_data['importes']['total_factura']}€")
        print(f"CNAE Emisor: {factura_data['emisor']['actividad_cnae']}")
        print(f"CNAE Receptor: {factura_data['receptor']['actividad_cnae']}")
        print()

        # Búsquedas específicas para consultoría informática
        specific_searches = [
            "servicios consultoría informática deducible",
            "desarrollo ERP gasto deducible",
            "6202 consultoría informática IVA",
            "servicios profesionales informática IRPF",
            "implementación software empresarial deducción",
            "consultoría tecnológica actividad empresarial",
            "desarrollo sistemas informáticos deducible",
            "servicios TI empresariales IVA deducible"
        ]

        all_results = []

        # Navegar al sitio
        scraper.navigate_to_search()

        # Realizar búsquedas específicas
        for search_term in specific_searches:
            print(f"🔍 Buscando: {search_term}")
            results = scraper.perform_search(search_term, "texto_libre")
            all_results.extend(results)
            print(f"   → Encontrados {len(results)} resultados")

        # Búsquedas adicionales por CNAE
        cnae_searches = ["6202", "6201", "consultoría informática"]
        for cnae in cnae_searches:
            print(f"🔍 Buscando CNAE/concepto: {cnae}")
            results = scraper.perform_search(cnae, "texto_libre")
            all_results.extend(results)

        print(f"\n📊 TOTAL DE CONSULTAS ENCONTRADAS: {len(all_results)}")

        if all_results:
            # Eliminar duplicados
            unique_results = []
            seen_titles = set()
            for result in all_results:
                if result['titulo'] not in seen_titles:
                    unique_results.append(result)
                    seen_titles.add(result['titulo'])

            print(f"📊 CONSULTAS ÚNICAS: {len(unique_results)}")

            # Guardar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            df = scraper.save_results(unique_results, f"consultoria_informatica_{timestamp}")

            # Análisis específico para consultoría informática
            benefits = analyze_consultoria_benefits(unique_results, factura_data)

            # Mostrar análisis
            print_analysis_results(benefits, unique_results)

            # Guardar análisis específico
            save_specific_analysis(benefits, factura_data, timestamp)

        else:
            print("❌ No se encontraron resultados")

    except Exception as e:
        print(f"❌ Error en análisis: {e}")
    finally:
        scraper.close()


def analyze_consultoria_benefits(results, factura_data):
    """Análisis específico para servicios de consultoría informática"""
    benefits = {
        'deducibilidad_servicios': [],
        'tratamiento_iva': [],
        'retenciones_irpf': [],
        'gastos_i_d': [],
        'amortizaciones': [],
        'incentivos_digitalizacion': [],
        'recomendaciones': []
    }

    importe = factura_data['importes']['total_factura']

    for result in results:
        contenido = result['contenido'].lower()
        titulo = result['titulo'].lower()

        # Deducibilidad de servicios
        if any(word in contenido for word in ['servicio', 'consultoría', 'deducible', 'gasto']):
            benefits['deducibilidad_servicios'].append({
                'consulta': result['titulo'],
                'relevancia': result['relevancia'],
                'aplicable': 'desarrollo ERP' in contenido or 'software' in contenido
            })

        # Tratamiento IVA
        if 'iva' in contenido and any(word in contenido for word in ['deducible', 'soportado', '21%']):
            benefits['tratamiento_iva'].append({
                'consulta': result['titulo'],
                'tipo': 'IVA Deducible',
                'tasa_aplicable': '21%'
            })

        # Retenciones IRPF
        if 'irpf' in contenido and any(word in contenido for word in ['15%', 'retencion', 'profesional']):
            benefits['retenciones_irpf'].append({
                'consulta': result['titulo'],
                'tipo': 'Retención IRPF',
                'tasa': '15%'
            })

        # I+D+i
        if any(word in contenido for word in ['i+d', 'innovacion', 'desarrollo', 'investigacion']):
            benefits['gastos_i_d'].append({
                'consulta': result['titulo'],
                'tipo': 'Posible I+D+i',
                'incentivo': 'Deducción adicional'
            })

        # Digitalización
        if any(word in contenido for word in ['digitalizacion', 'transformacion digital', 'tecnologia']):
            benefits['incentivos_digitalizacion'].append({
                'consulta': result['titulo'],
                'tipo': 'Kit Digital / Digitalización',
                'aplicable': importe <= 12000  # Límites típicos Kit Digital
            })

    # Generar recomendaciones específicas
    benefits['recomendaciones'] = generate_specific_recommendations(factura_data, benefits)

    return benefits


def generate_specific_recommendations(factura_data, benefits):
    """Genera recomendaciones específicas basadas en el análisis"""
    recommendations = []

    importe = factura_data['importes']['total_factura']
    base_imponible = factura_data['importes']['base_imponible']
    iva = factura_data['importes']['iva_21']
    irpf = factura_data['importes']['irpf']

    recommendations.append({
        'categoria': 'Deducibilidad',
        'recomendacion': f'El gasto de {base_imponible}€ por consultoría informática es DEDUCIBLE al 100% como gasto necesario para la actividad empresarial.',
        'base_legal': 'Art. 14 LIS - Gastos deducibles',
        'ahorro_estimado': base_imponible * 0.25  # Estimación IS 25%
    })

    recommendations.append({
        'categoria': 'IVA',
        'recomendacion': f'El IVA soportado de {iva}€ es DEDUCIBLE al 100% por tratarse de servicios utilizados exclusivamente para la actividad empresarial.',
        'base_legal': 'Art. 95 LIVA - IVA deducible',
        'ahorro_estimado': iva
    })

    recommendations.append({
        'categoria': 'IRPF',
        'recomendacion': f'La retención de IRPF de {irpf}€ (15%) es correcta para servicios profesionales informáticos.',
        'base_legal': 'Art. 95.1 RIRPF',
        'ahorro_estimado': 0  # Es una retención, no ahorro adicional
    })

    # Recomendación I+D si aplica
    if any('desarrollo' in concepto['descripcion'].lower() for concepto in factura_data['conceptos']):
        recommendations.append({
            'categoria': 'I+D+i',
            'recomendacion': 'EVALUAR si el desarrollo del ERP califica como I+D+i para deducción adicional del 25-42%.',
            'base_legal': 'Art. 35 LIS - Incentivos I+D+i',
            'ahorro_estimado': base_imponible * 0.25  # Deducción adicional estimada
        })

    # Kit Digital si aplica
    if factura_data['receptor']['tipo_empresa'] in ['SL', 'SA'] and importe <= 12000:
        recommendations.append({
            'categoria': 'Subvenciones',
            'recomendacion': 'VERIFICAR elegibilidad para Kit Digital (hasta 12.000€ para digitalización).',
            'base_legal': 'Plan de Digitalización PYME',
            'ahorro_estimado': min(importe * 0.8, 9600)  # Hasta 80% subvencionado
        })

    return recommendations


def print_analysis_results(benefits, results):
    """Muestra los resultados del análisis de forma estructurada"""
    print("\n" + "=" * 60)
    print("📋 ANÁLISIS DE BENEFICIOS FISCALES")
    print("=" * 60)

    # Top consultas más relevantes
    top_results = sorted(results, key=lambda x: x['relevancia'], reverse=True)[:5]
    print(f"\n🏆 TOP 5 CONSULTAS MÁS RELEVANTES:")
    for i, result in enumerate(top_results, 1):
        print(f"{i}. {result['titulo']} (⭐ {result['relevancia']})")
        print(f"   📁 {result['tipo']}")
        print(f"   📝 {result['contenido'][:100]}...")
        print()

    # Mostrar beneficios encontrados
    categories = [
        ('deducibilidad_servicios', '💼 DEDUCIBILIDAD DE SERVICIOS'),
        ('tratamiento_iva', '💰 TRATAMIENTO IVA'),
        ('retenciones_irpf', '📊 RETENCIONES IRPF'),
        ('gastos_i_d', '🔬 GASTOS I+D+i'),
        ('incentivos_digitalizacion', '💻 INCENTIVOS DIGITALIZACIÓN')
    ]

    for key, title in categories:
        items = benefits.get(key, [])
        if items:
            print(f"\n{title} ({len(items)} encontrados):")
            for item in items[:3]:  # Mostrar top 3
                print(f"  • {item.get('consulta', 'N/A')}")

    # Mostrar recomendaciones
    if benefits.get('recomendaciones'):
        print(f"\n💡 RECOMENDACIONES ESPECÍFICAS:")
        total_ahorro = 0
        for rec in benefits['recomendaciones']:
            print(f"\n📌 {rec['categoria'].upper()}")
            print(f"   {rec['recomendacion']}")
            print(f"   📜 Base legal: {rec['base_legal']}")
            if rec['ahorro_estimado'] > 0:
                print(f"   💵 Ahorro estimado: {rec['ahorro_estimado']:.2f}€")
                total_ahorro += rec['ahorro_estimado']

        print(f"\n💰 AHORRO FISCAL TOTAL ESTIMADO: {total_ahorro:.2f}€")


def save_specific_analysis(benefits, factura_data, timestamp):
    """Guarda análisis específico en JSON"""
    analysis_report = {
        'factura': factura_data['identificacion'],
        'fecha_analisis': datetime.now().isoformat(),
        'beneficios_identificados': benefits,
        'resumen_fiscal': {
            'base_imponible': factura_data['importes']['base_imponible'],
            'iva_deducible': factura_data['importes']['iva_21'],
            'retencion_irpf': factura_data['importes']['irpf']
        }
    }

    filename = f"analisis_fiscal_detallado_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Análisis detallado guardado en: {filename}")


if __name__ == "__main__":
    analyze_factura_consultoria()