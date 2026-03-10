#!/usr/bin/env python3
"""
Script para realizar benchmarking de APIs con consultas secuenciales y paralelas
"""

import requests
import time
import statistics
import asyncio
import aiohttp
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class APIBenchmark:
    def __init__(self):
        # Configuración de las APIs
        self.fiscal_url = "http://212.69.86.224:8001/fiscal/classify"
        # Timeout para las peticiones (en segundos)
        self.timeout = 120
        # Número de consultas secuenciales a realizar
        self.sequential_queries = 3000
        
        self.fiscal_payload = {
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
                "retencion_aplicada": True,
                "tipo_retencion": 15,
                "operacion_intracomunitaria": False
            },
            "relacion_comercial": {
                "tipo_relacion": "Tercero independiente",
                "empresa_vinculada": False,
                "operacion_vinculada": False,
                "proveedor_habitual": True
            },
            "fechas": {
                "emision": "2024-10-15",
                "prestacion_servicio": "2024-10-01",
                "pago_efectivo": "2024-10-20"
            }
        }
        
        self.headers = {"Content-Type": "application/json"}

    def make_chat_request(self) -> tuple[float, bool, str]:
        """Realiza una consulta a la API de chat"""
        start_time = time.time()
        try:
            response = requests.post(
                self.chat_url, 
                json=self.chat_payload, 
                headers=self.headers,
                timeout=self.timeout
            )
            end_time = time.time()
            duration = end_time - start_time
            success = response.status_code == 200
            if success:
                message = f"Status: {response.status_code}"
            else:
                # Incluir el cuerpo de la respuesta cuando no sea 200
                body_preview = (response.text or "").strip()
                if len(body_preview) > 800:
                    body_preview = body_preview[:800] + "..."
                message = f"Status: {response.status_code} - Body: {body_preview}"
            return duration, success, message
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            return duration, False, f"Error: {str(e)}"

    def make_fiscal_request(self) -> tuple[float, bool, str]:
        """Realiza una consulta a la API fiscal"""
        start_time = time.time()
        try:
            response = requests.post(
                self.fiscal_url, 
                json=self.fiscal_payload, 
                headers=self.headers,
                timeout=self.timeout
            )
            end_time = time.time()
            duration = end_time - start_time
            success = response.status_code == 200
            if success:
                message = f"Status: {response.status_code}"
            else:
                # Incluir el cuerpo de la respuesta cuando no sea 200
                body_preview = (response.text or "").strip()
                if len(body_preview) > 800:
                    body_preview = body_preview[:800] + "..."
                message = f"Status: {response.status_code} - Body: {body_preview}"
            return duration, success, message
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            return duration, False, f"Error: {str(e)}"

    def calculate_stats(self, times: List[float]) -> Dict[str, float]:
        """Calcula estadísticas de los tiempos"""
        if not times:
            return {"promedio": 0, "max": 0, "min": 0, "std": 0}
        
        return {
            "promedio": statistics.mean(times),
            "max": max(times),
            "min": min(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0
        }

    def print_stats(self, title: str, times: List[float], successes: List[bool]):
        """Imprime estadísticas formateadas"""
        stats = self.calculate_stats(times)
        success_rate = sum(successes) / len(successes) * 100 if successes else 0
        
        print(f"\n{'='*50}")
        print(f"{title}")
        print(f"{'='*50}")
        print(f"Total de consultas: {len(times)}")
        print(f"Consultas exitosas: {sum(successes)}/{len(successes)} ({success_rate:.1f}%)")
        print(f"Tiempo promedio: {stats['promedio']:.3f}s")
        print(f"Tiempo máximo: {stats['max']:.3f}s")
        print(f"Tiempo mínimo: {stats['min']:.3f}s")
        print(f"Desviación estándar: {stats['std']:.3f}s")

    def run_sequential_tests(self):
        """Ejecuta 30 consultas secuenciales para cada API"""
        print("🔄 Iniciando pruebas secuenciales...")
        
        # Pruebas para API Fiscal
        print(f"\n💰 Probando API Fiscal ({self.sequential_queries} consultas secuenciales)...")
        fiscal_times = []
        fiscal_successes = []
        
        for i in range(self.sequential_queries):
            print(f"Fiscal consulta {i+1}/{self.sequential_queries}...", end=" ")
            duration, success, message = self.make_fiscal_request()
            fiscal_times.append(duration)
            fiscal_successes.append(success)
            print(f"{duration:.3f}s - {message}")
        
        self.print_stats("RESULTADOS SECUENCIALES - API FISCAL", fiscal_times, fiscal_successes)

    def run_parallel_tests(self):
        """Ejecuta 10 consultas paralelas para cada API"""
        print("\n🚀 Iniciando pruebas paralelas...")
        
        # Pruebas paralelas para API Fiscal
        print("\n💰 Probando API Fiscal (10 consultas paralelas)...")
        fiscal_times = []
        fiscal_successes = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self.make_fiscal_request) for _ in range(10)]
            for i, future in enumerate(as_completed(futures)):
                duration, success, message = future.result()
                fiscal_times.append(duration)
                fiscal_successes.append(success)
                print(f"Fiscal paralela {i+1}/10: {duration:.3f}s - {message}")
        
        self.print_stats("RESULTADOS PARALELOS - API FISCAL", fiscal_times, fiscal_successes)

    def run_full_benchmark(self):
        """Ejecuta el benchmark completo"""
        print("🎯 Iniciando Benchmark de APIs")
        print("🔗 URLs de prueba:")
        print(f"  - Fiscal API: {self.fiscal_url}")
        
        start_total = time.time()
        
        # Ejecutar pruebas secuenciales
        self.run_sequential_tests()
        
        # Ejecutar pruebas paralelas
        self.run_parallel_tests()
        
        end_total = time.time()
        total_duration = end_total - start_total
        
        print(f"\n🏁 Benchmark completado en {total_duration:.2f} segundos")

def main():
    """Función principal"""
    try:
        benchmark = APIBenchmark()
        benchmark.run_full_benchmark()
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante el benchmark: {e}")

if __name__ == "__main__":
    main()
