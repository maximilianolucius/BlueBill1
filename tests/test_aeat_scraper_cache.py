#!/usr/bin/env python3
"""
Prueba de funcionamiento del sistema de caché de AEATConsultaScraper (aeat_scraper.py)

Validaciones que realiza:
- Sembrado manual de resultados en cache persistente (SQLite) y verificación de hit.
- Comprobación de cache en memoria (segunda lectura más rápida).
- Estadísticas básicas del cache.
- Prueba de expiración creando una cache con duración 0 meses.

Nota: No accede a la web de la AEAT; usa el cache del scraper.
"""

import os
import time
import shutil
import tempfile
from pathlib import Path

from aeat_scraper import AEATConsultaScraper


def main():
    print("=== Test de caché AEATConsultaScraper ===")

    # Directorio temporal aislado para el cache
    base_tmp = Path(tempfile.gettempdir()) / "aeat_cache_test"
    cache_dir = base_tmp / "cache"
    shutil.rmtree(base_tmp, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Instanciar scraper apuntando a nuestro cache temporal
    scraper = AEATConsultaScraper(
        headless=True,
        verbose=True,
        cache_dir=str(cache_dir),
        cache_duration_months=3,  # 3 meses
        use_pool=False,
        pool_size=2,
    )

    # Query de prueba y resultados falsos
    query = "prueba_cache_deducible_iva"
    fake_results = [
        {
            "numero_consulta": "V0000-00",
            "cuestion_planteada": "Prueba de caché - IVA deducible",
            "contestacion_completa": "Contenido de ejemplo para validar caché.",
            "relevancia": 0.99,
        }
    ]

    # 1) Sembrar cache y validar HIT desde persistencia
    scraper.cache.set(query, "texto_libre", fake_results)
    print("[1] Cache sembrado en SQLite y memoria")

    t0 = time.perf_counter()
    hit_results_1 = scraper.perform_search(query, "texto_libre")
    t1 = time.perf_counter() - t0

    assert hit_results_1 == fake_results, "Resultados desde cache no coinciden"
    print(f"[1] HIT desde cache persistente OK (len={len(hit_results_1)}), tiempo: {t1*1000:.1f} ms")

    # 2) Segunda lectura (debería venir del cache en memoria) aún más rápida
    t0 = time.perf_counter()
    hit_results_2 = scraper.perform_search(query, "texto_libre")
    t2 = time.perf_counter() - t0
    assert hit_results_2 == fake_results, "Resultados desde cache (memoria) no coinciden"
    print(f"[2] HIT desde cache en memoria OK (len={len(hit_results_2)}), tiempo: {t2*1000:.1f} ms")

    # 3) Estadísticas del cache
    stats = scraper.cache.get_cache_stats()
    print("[3] Estadísticas del cache:")
    print(f"    - Entradas totales: {stats['total_entries']}")
    print(f"    - Hits totales: {stats['total_hits']}")
    print(f"    - En memoria: {stats['memory_cache_size']}")

    # 4) Prueba de expiración (cache con duración 0 meses -> expira inmediato)
    exp_dir = base_tmp / "cache_expira"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_scraper = AEATConsultaScraper(
        headless=True,
        verbose=False,
        cache_dir=str(exp_dir),
        cache_duration_months=0,  # expira en el mismo instante
        use_pool=False,
    )
    exp_scraper.cache.set("q_expira", "texto_libre", fake_results)
    expired = exp_scraper.cache.get("q_expira", "texto_libre")
    assert expired is None, "La entrada debería haber expirado inmediatamente"
    print("[4] Expiración inmediata validada (cache_duration_months=0)")

    # Limpieza
    shutil.rmtree(base_tmp, ignore_errors=True)
    print("✔ Test de caché completado correctamente")


if __name__ == "__main__":
    main()

