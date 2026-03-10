#!/usr/bin/env python3
"""
Módulo de web scraping para consultas tributarias de la AEAT.

Proporciona herramientas para automatizar la búsqueda y extracción de consultas
tributarias (generales y vinculantes) del portal de la Agencia Estatal de
Administración Tributaria (AEAT), basándose en los datos de facturas.

Incluye:
    - AEATSearchCache: Sistema de caché persistente en SQLite con caché en memoria.
    - DriverPool: Pool de drivers Selenium para ejecución concurrente de búsquedas.
    - AEATConsultaScraper: Scraper principal que orquesta la navegación, búsqueda,
      extracción de resultados y análisis de beneficios fiscales.

Dependencias externas:
    - Selenium con ChromeDriver para la automatización del navegador.
    - BeautifulSoup y requests para procesamiento HTML auxiliar.
    - pandas para la exportación de resultados a Excel.
"""

import json
import time
import traceback

import pandas as pd
import uuid
import os
import tempfile
import random
import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import requests
import re
from utils.generar_keywords_y_fiscal_terms import generar_keywords_y_fiscal_terms


class AEATSearchCache:
    """Sistema de caché persistente en SQLite para consultas AEAT.

    Implementa un caché de dos niveles (memoria y disco) para almacenar
    los resultados de búsquedas en el portal de la AEAT, evitando consultas
    redundantes y mejorando el rendimiento.

    Attributes:
        cache_duration_months (int): Duración en meses antes de que una entrada expire.
        cache_dir (Path): Ruta al directorio donde se almacena la base de datos del caché.
        db_path (Path): Ruta completa al archivo SQLite del caché.
    """

    def __init__(self, cache_dir=None, cache_duration_months=3):
        """
        Inicializa el sistema de cache

        Args:
            cache_dir: Directorio donde almacenar el cache (por defecto: ~/.aeat_cache)
            cache_duration_months: Duración del cache en meses (por defecto: 3)
        """
        self.cache_duration_months = cache_duration_months

        # Directorio de cache
        if cache_dir is None:
            self.cache_dir = Path.home() / ".aeat_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(exist_ok=True)

        # Archivo de base de datos
        self.db_path = self.cache_dir / "aeat_search_cache.db"

        # Inicializar base de datos
        self._init_database()

        # Cache en memoria para optimizar accesos frecuentes
        self._memory_cache = {}

    def _init_database(self):
        """Inicializa la base de datos SQLite creando la tabla y los índices necesarios."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query_hash TEXT PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    search_type TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    hits INTEGER DEFAULT 0
                )
            """)

            # Índice para mejorar performance en consultas por fecha
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at 
                ON search_cache(expires_at)
            """)

            conn.commit()

    def _generate_cache_key(self, search_query, search_type):
        """Genera una clave hash SHA-256 única para identificar la consulta en el caché.

        Args:
            search_query (str): Texto de la consulta de búsqueda.
            search_type (str): Tipo de búsqueda (p.ej. 'texto_libre', 'normativa').

        Returns:
            str: Hash SHA-256 hexadecimal que identifica unívocamente la consulta.
        """
        key_data = f"{search_query.strip().lower()}|{search_type}"
        return hashlib.sha256(key_data.encode('utf-8')).hexdigest()

    def _is_expired(self, expires_at_str):
        """Verifica si un elemento del caché ha expirado comparando con la fecha actual.

        Args:
            expires_at_str (str): Fecha de expiración en formato ISO 8601.

        Returns:
            bool: True si la entrada ha expirado, False en caso contrario.
        """
        expires_at = datetime.fromisoformat(expires_at_str)
        return datetime.now() > expires_at

    def get(self, search_query, search_type="texto_libre"):
        """Obtiene resultados del caché si existen y no han expirado.

        Busca primero en el caché en memoria y, si no encuentra, consulta la
        base de datos SQLite. Incrementa el contador de hits en caso de acierto.

        Args:
            search_query (str): Texto de la consulta de búsqueda.
            search_type (str): Tipo de búsqueda (por defecto: 'texto_libre').

        Returns:
            list o None: Lista de resultados cacheados, o None si la entrada
            no existe o ha expirado.
        """
        cache_key = self._generate_cache_key(search_query, search_type)

        # Primero verificar cache en memoria
        if cache_key in self._memory_cache:
            cached_item = self._memory_cache[cache_key]
            if not self._is_expired(cached_item['expires_at']):
                return cached_item['results']
            else:
                # Eliminar del cache en memoria si expiró
                del self._memory_cache[cache_key]

        # Consultar base de datos
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT query_text, results_json, expires_at, hits
                FROM search_cache 
                WHERE query_hash = ?
            """, (cache_key,))

            row = cursor.fetchone()

            if row is None:
                return None

            # Verificar si expiró
            if self._is_expired(row['expires_at']):
                # Eliminar entrada expirada
                conn.execute("DELETE FROM search_cache WHERE query_hash = ?", (cache_key,))
                conn.commit()
                return None

            # Incrementar contador de hits
            conn.execute("""
                UPDATE search_cache 
                SET hits = hits + 1 
                WHERE query_hash = ?
            """, (cache_key,))
            conn.commit()

            # Deserializar resultados
            results = json.loads(row['results_json'])

            # Guardar en cache en memoria
            self._memory_cache[cache_key] = {
                'results': results,
                'expires_at': row['expires_at']
            }

            return results

    def set(self, search_query, search_type, results):
        """Almacena resultados en el caché (memoria y base de datos).

        Calcula la fecha de expiración según ``cache_duration_months`` y persiste
        los resultados tanto en SQLite como en el diccionario en memoria.

        Args:
            search_query (str): Texto de la consulta de búsqueda.
            search_type (str): Tipo de búsqueda (p.ej. 'texto_libre', 'normativa').
            results (list): Lista de resultados a almacenar en el caché.
        """
        cache_key = self._generate_cache_key(search_query, search_type)

        # Calcular fecha de expiración
        created_at = datetime.now()
        expires_at = created_at + timedelta(days=30 * self.cache_duration_months)

        # Serializar resultados
        results_json = json.dumps(results, ensure_ascii=False)

        # Guardar en base de datos
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO search_cache 
                (query_hash, query_text, search_type, results_json, created_at, expires_at, hits)
                VALUES (?, ?, ?, ?, ?, ?, 0)
            """, (
                cache_key,
                search_query,
                search_type,
                results_json,
                created_at.isoformat(),
                expires_at.isoformat()
            ))
            conn.commit()

        # Guardar en cache en memoria
        self._memory_cache[cache_key] = {
            'results': results,
            'expires_at': expires_at.isoformat()
        }

    def cleanup_expired(self):
        """Elimina entradas expiradas del caché (base de datos y memoria).

        Returns:
            int: Número de entradas eliminadas de la base de datos.
        """
        current_time = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM search_cache 
                WHERE expires_at < ?
            """, (current_time,))

            deleted_count = cursor.rowcount
            conn.commit()

        # Limpiar cache en memoria
        expired_keys = []
        for key, item in self._memory_cache.items():
            if self._is_expired(item['expires_at']):
                expired_keys.append(key)

        for key in expired_keys:
            del self._memory_cache[key]

        return deleted_count

    def get_cache_stats(self):
        """Obtiene estadísticas detalladas del caché.

        Returns:
            dict: Diccionario con las claves: 'total_entries', 'total_hits',
            'avg_hits_per_entry', 'oldest_entry', 'newest_entry',
            'entries_by_type', 'expired_entries', 'memory_cache_size'.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Estadísticas generales
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(hits) as total_hits,
                    AVG(hits) as avg_hits_per_entry,
                    MIN(created_at) as oldest_entry,
                    MAX(created_at) as newest_entry
                FROM search_cache
            """)

            general_stats = cursor.fetchone()

            # Entradas por tipo
            cursor = conn.execute("""
                SELECT search_type, COUNT(*) as count
                FROM search_cache
                GROUP BY search_type
            """)

            type_stats = cursor.fetchall()

            # Entradas expiradas
            current_time = datetime.now().isoformat()
            cursor = conn.execute("""
                SELECT COUNT(*) as expired_count
                FROM search_cache
                WHERE expires_at < ?
            """, (current_time,))

            expired_stats = cursor.fetchone()

            return {
                'total_entries': general_stats['total_entries'],
                'total_hits': general_stats['total_hits'],
                'avg_hits_per_entry': round(general_stats['avg_hits_per_entry'] or 0, 2),
                'oldest_entry': general_stats['oldest_entry'],
                'newest_entry': general_stats['newest_entry'],
                'entries_by_type': dict(type_stats),
                'expired_entries': expired_stats['expired_count'],
                'memory_cache_size': len(self._memory_cache)
            }

    def clear_all(self):
        """Limpia completamente el caché, eliminando todas las entradas de la base de datos y la memoria."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM search_cache")
            conn.commit()

        self._memory_cache.clear()


class DriverPool:
    """Pool de instancias de ChromeDriver de Selenium para ejecución paralela.

    Gestiona la creación, asignación y liberación de múltiples drivers de Chrome,
    cada uno con su propio directorio temporal y puerto de depuración, permitiendo
    la ejecución concurrente de búsquedas en el portal de la AEAT.

    Attributes:
        pool_size (int): Número total de drivers en el pool.
        headless (bool): Si True, los navegadores se ejecutan sin interfaz gráfica.
        verbose (bool): Si True, se imprimen mensajes de log detallados.
        pool_id (str): Identificador único del pool (8 caracteres UUID).
        available_drivers (list): Lista de drivers actualmente disponibles.
        busy_drivers (set): Conjunto de IDs de drivers actualmente en uso.
        driver_lock (threading.Lock): Lock para sincronizar el acceso al pool.
        pool_temp_dir (Path): Directorio temporal del pool para datos de Chrome.
    """

    def __init__(self, pool_size=3, headless=True, verbose=True):
        """Inicializa el pool de drivers de Selenium.

        Args:
            pool_size (int): Número de drivers en el pool (por defecto: 3).
            headless (bool): Ejecutar en modo headless (por defecto: True).
            verbose (bool): Mostrar logs detallados (por defecto: True).
        """
        self.pool_size = pool_size
        self.headless = headless
        self.verbose = verbose
        self.pool_id = str(uuid.uuid4())[:8]

        # Pool de drivers disponibles
        self.available_drivers = []
        self.busy_drivers = set()
        self.driver_lock = threading.Lock()

        # Crear directorio temporal para el pool
        self.pool_temp_dir = Path(tempfile.gettempdir()) / f"aeat_pool_{self.pool_id}"
        self.pool_temp_dir.mkdir(exist_ok=True)

        self._log(f"Inicializando pool de {pool_size} drivers...")
        self._init_drivers()

    def _log(self, message):
        """Imprime un mensaje de log con el identificador del pool como prefijo.

        Args:
            message (str): Mensaje a imprimir.
        """
        if self.verbose:
            print(f"[POOL-{self.pool_id}] {message}")

    def _error(self, message):
        """Imprime un mensaje de error con el identificador del pool como prefijo.

        Args:
            message (str): Mensaje de error a imprimir.
        """
        print(f"[POOL-{self.pool_id}] ERROR: {message}")

    def _create_driver(self, driver_id):
        """Crea un driver de Chrome individual con configuración aislada.

        Cada driver recibe su propio directorio de datos de usuario, puerto de
        depuración remota y directorio de caché para evitar conflictos entre
        instancias concurrentes.

        Args:
            driver_id (str): Identificador único para este driver.

        Returns:
            webdriver.Chrome o None: Instancia del driver configurado, o None
            si ocurre un error durante la creación.
        """
        chrome_options = Options()

        # Configuración básica
        if self.headless:
            chrome_options.add_argument("--headless")

        # Configuraciones para evitar conflictos
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--window-size=1920,1080")

        # Directorio de datos único para cada driver
        driver_temp_dir = self.pool_temp_dir / f"driver_{driver_id}"
        driver_temp_dir.mkdir(exist_ok=True)

        user_data_dir = driver_temp_dir / "chrome_user_data"
        chrome_options.add_argument(f"--user-data-dir={user_data_dir}")

        # Puerto de debugging único
        debug_port = random.randint(9222, 9999)
        chrome_options.add_argument(f"--remote-debugging-port={debug_port}")

        # Directorio de cache único
        cache_dir = driver_temp_dir / "chrome_cache"
        chrome_options.add_argument(f"--disk-cache-dir={cache_dir}")

        # Configuraciones adicionales para estabilidad
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-client-side-phishing-detection")
        chrome_options.add_argument("--disable-crash-reporter")
        chrome_options.add_argument("--disable-oopr-debug-crash-dump")
        chrome_options.add_argument("--no-crash-upload")
        chrome_options.add_argument("--disable-low-res-tiling")
        chrome_options.add_argument("--disable-default-apps")

        # Configurar prefs
        prefs = {
            "profile.default_content_setting_values": {
                "notifications": 2,
                "popups": 2,
                "geolocation": 2,
                "media_stream": 2,
            },
            "profile.default_content_settings": {"popups": 0},
            "profile.managed_default_content_settings": {"images": 2}
        }
        chrome_options.add_experimental_option("prefs", prefs)

        try:
            driver = webdriver.Chrome(options=chrome_options)
            driver.driver_id = driver_id
            driver.temp_dir = driver_temp_dir
            return driver
        except Exception as e:
            self._error(f"Error creando driver {driver_id}: {e}")
            return None

    def _init_drivers(self):
        """Inicializa todos los drivers del pool creándolos secuencialmente."""
        for i in range(self.pool_size):
            driver_id = f"{self.pool_id}_d{i}"
            driver = self._create_driver(driver_id)
            if driver:
                self.available_drivers.append(driver)
                self._log(f"Driver {driver_id} creado correctamente")
            else:
                self._error(f"Falló creación del driver {driver_id}")

    def get_driver(self):
        """Obtiene un driver disponible del pool de forma thread-safe.

        Returns:
            webdriver.Chrome o None: Un driver libre del pool, o None si no
            hay drivers disponibles en ese momento.
        """
        with self.driver_lock:
            if not self.available_drivers:
                return None

            driver = self.available_drivers.pop()
            self.busy_drivers.add(driver.driver_id)
            return driver

    def return_driver(self, driver):
        """Devuelve un driver al pool marcándolo como disponible.

        Args:
            driver (webdriver.Chrome): Driver previamente obtenido con ``get_driver()``.
        """
        with self.driver_lock:
            if driver.driver_id in self.busy_drivers:
                self.busy_drivers.remove(driver.driver_id)
                self.available_drivers.append(driver)

    def get_pool_status(self):
        """Obtiene el estado actual del pool de drivers.

        Returns:
            dict: Diccionario con 'total_drivers', 'available_drivers',
            'busy_drivers' y 'pool_id'.
        """
        with self.driver_lock:
            return {
                'total_drivers': self.pool_size,
                'available_drivers': len(self.available_drivers),
                'busy_drivers': len(self.busy_drivers),
                'pool_id': self.pool_id
            }

    def close_all(self):
        """Cierra todos los drivers disponibles del pool y limpia el directorio temporal."""
        with self.driver_lock:
            # Cerrar drivers disponibles
            for driver in self.available_drivers:
                try:
                    driver.quit()
                    self._log(f"Driver {driver.driver_id} cerrado")
                except Exception as e:
                    self._error(f"Error cerrando driver {driver.driver_id}: {e}")

            # Los drivers busy se cerrarán cuando sean devueltos
            self.available_drivers.clear()

        # Limpiar directorio temporal del pool
        try:
            import shutil
            if self.pool_temp_dir.exists():
                shutil.rmtree(self.pool_temp_dir)
                self._log("Directorio temporal del pool limpiado")
        except Exception as e:
            self._error(f"Error limpiando directorio temporal: {e}")


class AEATConsultaScraper:
    """Scraper principal para consultas tributarias del portal de la AEAT.

    Orquesta la navegación web, búsqueda de consultas generales y vinculantes,
    extracción de resultados, almacenamiento en caché y análisis de beneficios
    fiscales. Soporta tanto ejecución individual (un solo driver) como ejecución
    paralela mediante un pool de drivers.

    Attributes:
        base_url (str): URL base del portal de consultas de la AEAT.
        verbose (bool): Si True, se imprimen mensajes de log.
        use_pool (bool): Si True, utiliza un pool de drivers para paralelismo.
        cache (AEATSearchCache): Instancia del sistema de caché persistente.
        instance_id (str): Identificador único de 8 caracteres para esta instancia.
        session_timestamp (str): Marca de tiempo de la sesión para nombres de archivo.
        driver_pool (DriverPool o None): Pool de drivers (solo si use_pool=True).
        driver (webdriver.Chrome o None): Driver individual (solo si use_pool=False).
        wait (WebDriverWait o None): Objeto de espera asociado al driver individual.
    """

    def __init__(self, headless=True, verbose=True, cache_dir=None, cache_duration_months=3,
                 use_pool=False, pool_size=3):
        """Inicializa el scraper con la configuración de Selenium y caché.

        Args:
            headless (bool): Ejecutar el navegador sin interfaz gráfica (por defecto: True).
            verbose (bool): Mostrar mensajes de log detallados (por defecto: True).
            cache_dir (str o None): Directorio para el caché SQLite (por defecto: ~/.aeat_cache).
            cache_duration_months (int): Duración del caché en meses (por defecto: 3).
            use_pool (bool): Usar pool de drivers para ejecución paralela (por defecto: False).
            pool_size (int): Número de drivers en el pool (por defecto: 3).

        Raises:
            Exception: Si falla la creación del driver de Chrome en modo individual.
        """
        self.base_url = "https://petete.tributos.hacienda.gob.es/consultas/"
        self.verbose = verbose
        self.use_pool = use_pool

        # Inicializar sistema de cache
        self.cache = AEATSearchCache(cache_dir, cache_duration_months)

        # Generar ID único para esta instancia
        self.instance_id = str(uuid.uuid4())[:8]
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if use_pool:
            # Usar pool de drivers para ejecución paralela
            self.driver_pool = DriverPool(pool_size, headless, verbose)
            self.driver = None  # No hay driver individual
            self.wait = None
            self._log(f"Instancia {self.instance_id} usando pool de {pool_size} drivers")
        else:
            # Usar driver individual (modo original)
            self.driver_pool = None
            self.driver = None

            # Crear directorio temporal único para esta instancia
            self.temp_dir = Path(tempfile.gettempdir()) / f"aeat_scraper_{self.instance_id}"
            self.temp_dir.mkdir(exist_ok=True)

            self._log(f"Iniciando instancia individual {self.instance_id}")
            self.setup_driver(headless)

    def _log(self, message):
        """Imprime un mensaje de log con el identificador de instancia, solo si verbose=True.

        Args:
            message (str): Mensaje a imprimir.
        """
        if self.verbose:
            print(f"[{self.instance_id}] {message}")

    def _error(self, message):
        """Imprime un mensaje de error con el identificador de instancia (siempre visible).

        Args:
            message (str): Mensaje de error a imprimir.
        """
        print(f"[{self.instance_id}] ERROR: {message}")

    def setup_driver(self, headless=True):
        """Configura el driver de Chrome con directorios y puertos únicos por instancia.

        Solo se ejecuta en modo individual (use_pool=False). Cada instancia recibe
        su propio directorio de datos de usuario, caché y puerto de depuración remota.

        Args:
            headless (bool): Ejecutar el navegador sin interfaz gráfica (por defecto: True).

        Raises:
            Exception: Si falla la creación o configuración del driver de Chrome.
        """
        if self.use_pool:
            return  # No configurar driver individual si se usa pool

        chrome_options = Options()

        # Configuración básica
        if headless:
            chrome_options.add_argument("--headless")

        # Configuraciones para evitar conflictos entre instancias
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--disable-images")
        chrome_options.add_argument("--disable-javascript")
        chrome_options.add_argument("--window-size=1920,1080")

        # Directorio de datos único para cada instancia
        user_data_dir = self.temp_dir / "chrome_user_data"
        chrome_options.add_argument(f"--user-data-dir={user_data_dir}")

        # Puerto de debugging único para cada instancia
        debug_port = random.randint(9222, 9999)
        chrome_options.add_argument(f"--remote-debugging-port={debug_port}")

        # Directorio de cache único
        cache_dir = self.temp_dir / "chrome_cache"
        chrome_options.add_argument(f"--disk-cache-dir={cache_dir}")

        # Configuraciones adicionales para estabilidad
        chrome_options.add_argument("--disable-background-timer-throttling")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-client-side-phishing-detection")
        chrome_options.add_argument("--disable-crash-reporter")
        chrome_options.add_argument("--disable-oopr-debug-crash-dump")
        chrome_options.add_argument("--no-crash-upload")
        chrome_options.add_argument("--disable-low-res-tiling")
        chrome_options.add_argument("--disable-default-apps")

        # Configurar prefs para evitar diálogos y notificaciones
        prefs = {
            "profile.default_content_setting_values": {
                "notifications": 2,
                "popups": 2,
                "geolocation": 2,
                "media_stream": 2,
            },
            "profile.default_content_settings": {"popups": 0},
            "profile.managed_default_content_settings": {"images": 2}
        }
        chrome_options.add_experimental_option("prefs", prefs)

        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, 10)
            self._log("Driver configurado correctamente")
        except Exception as e:
            self._error(f"Error configurando driver: {e}")
            raise

    def load_factura_data(self, factura_path):
        """Carga los datos de una factura desde un archivo JSON.

        Args:
            factura_path (str): Ruta al archivo JSON con los datos de la factura.

        Returns:
            dict: Diccionario con los datos de la factura deserializados.
        """
        with open(factura_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_search_terms(self, factura_data, keywords):
        """Extrae términos de búsqueda relevantes de los datos de una factura.

        Analiza los campos de la factura (códigos CNAE, conceptos, información
        fiscal) y combina con las palabras clave proporcionadas.

        Args:
            factura_data (dict): Datos de la factura con claves como 'receptor',
                'emisor', 'conceptos' y 'fiscal'.
            keywords (list): Lista de palabras clave adicionales para la búsqueda.

        Returns:
            dict: Diccionario con las claves 'cnae_codes', 'conceptos',
            'impuestos' y 'palabras_clave', cada una conteniendo una lista
            de términos relevantes.
        """
        terms = {
            'cnae_codes': [],
            'conceptos': [],
            'impuestos': [],
            'palabras_clave': []
        }

        # CNAE codes
        if 'receptor' in factura_data and 'actividad_cnae' in factura_data['receptor']:
            terms['cnae_codes'].append(factura_data['receptor']['actividad_cnae'])
        if 'emisor' in factura_data and 'actividad_cnae' in factura_data['emisor']:
            terms['cnae_codes'].append(factura_data['emisor']['actividad_cnae'])

        # Conceptos y servicios
        if 'conceptos' in factura_data:
            for concepto in factura_data['conceptos']:
                if 'descripcion' in concepto:
                    terms['conceptos'].append(concepto['descripcion'])

        # Información fiscal
        if 'fiscal' in factura_data:
            fiscal = factura_data['fiscal']
            if fiscal.get('retencion_aplicada'):
                terms['impuestos'].append(f"IRPF {fiscal.get('tipo_retencion', '')}%")
            if fiscal.get('regimen_iva'):
                terms['impuestos'].append(f"IVA {fiscal['regimen_iva']}")

        # Palabras clave específicas
        terms['palabras_clave'] = keywords

        return terms

    def navigate_to_search(self, driver=None, wait=None):
        """Navega al formulario de búsqueda del portal de consultas de la AEAT.

        Carga la página base y espera a que el formulario de búsqueda esté
        disponible en el DOM.

        Args:
            driver (webdriver.Chrome o None): Driver específico a utilizar.
                Si es None, usa el driver de la instancia.
            wait (WebDriverWait o None): Objeto de espera asociado al driver.
                Si es None, usa el de la instancia.

        Raises:
            ValueError: Si no hay ningún driver disponible.
            Exception: Si falla la carga de la página o la espera del formulario.
        """
        # Usar driver específico o el driver de la instancia
        current_driver = driver or self.driver
        current_wait = wait or self.wait

        if not current_driver:
            raise ValueError("No hay driver disponible")

        self._log("Cargando página de consultas AEAT...")
        try:
            current_driver.get(self.base_url)

            # Esperar a que el formulario se cargue
            current_wait.until(
                EC.presence_of_element_located((By.ID, "knosys_form"))
            )
            time.sleep(3)  # Esperar carga completa del JavaScript
            self._log("Formulario cargado correctamente")
        except Exception as e:
            self._error(f"Error cargando formulario: {e}")
            raise

    def perform_search(self, search_query, search_type="texto_libre", driver=None, wait=None):
        """Realiza una búsqueda en el portal de la AEAT con soporte de caché.

        Primero intenta obtener los resultados del caché. Si no existen,
        ejecuta la búsqueda en el portal web rellenando el formulario según
        el tipo de búsqueda y extrae los resultados de las pestañas de
        Consultas Generales y Vinculantes.

        Args:
            search_query (str): Texto de la consulta a buscar.
            search_type (str): Tipo de búsqueda: 'texto_libre' (por defecto),
                'cuestion_planteada' o 'normativa'.
            driver (webdriver.Chrome o None): Driver específico (para uso con pool).
            wait (WebDriverWait o None): Objeto de espera asociado al driver.

        Returns:
            list: Lista de diccionarios con los resultados de la búsqueda.

        Raises:
            ValueError: Si no hay ningún driver disponible para la búsqueda.
        """

        # Intentar obtener resultados del cache primero
        cached_results = self.cache.get(search_query, search_type)
        if cached_results is not None:
            worker_id = getattr(driver, 'driver_id', 'main') if driver else 'main'
            self._log(
                f"[{worker_id}] Resultados obtenidos del cache para: '{search_query}' ({len(cached_results)} resultados)")
            return cached_results

        # Usar driver específico o el driver de la instancia
        current_driver = driver or self.driver
        current_wait = wait or self.wait
        worker_id = getattr(current_driver, 'driver_id', 'main') if current_driver else 'main'

        if not current_driver:
            raise ValueError("No hay driver disponible para la búsqueda")

        self._log(f"[{worker_id}] Cache miss - realizando búsqueda web para: '{search_query}'")
        results = []

        try:
            # Limpiar formulario
            self.clear_form(current_driver)

            # Seleccionar tipos de consulta
            consultas_generales = current_driver.find_element(By.ID, "generales")
            if not consultas_generales.is_selected():
                consultas_generales.click()

            consultas_vinculantes = current_driver.find_element(By.ID, "vinculantes")
            if not consultas_vinculantes.is_selected():
                consultas_vinculantes.click()

            # Introducir términos de búsqueda
            if search_type == "texto_libre":
                texto_libre = current_wait.until(
                    EC.presence_of_element_located((By.ID, "VLCMP_6"))
                )
                texto_libre.clear()
                texto_libre.send_keys(search_query)

            elif search_type == "cuestion_planteada":
                cuestion = current_wait.until(
                    EC.presence_of_element_located((By.ID, "VLCMP_4"))
                )
                cuestion.clear()
                cuestion.send_keys(search_query)

            elif search_type == "normativa":
                normativa = current_wait.until(
                    EC.presence_of_element_located((By.ID, "VLCMP_3"))
                )
                normativa.clear()
                normativa.send_keys(search_query)

            order_select = Select(current_driver.find_element(By.ID, "cmpOrder"))
            order_select.select_by_value("FECHA-SALIDA")

            # Seleccionar orden descendente
            dir_select = Select(current_driver.find_element(By.ID, "dirOrder"))
            dir_select.select_by_visible_text("Descendente")

            # Hacer clic en buscar
            buscar_btn = current_driver.find_element(By.XPATH, "//button[@type='submit' and text()='Buscar']")
            buscar_btn.click()

            # Esperar resultados
            time.sleep(3)

            # Extraer resultados de ambas pestañas
            results.extend(self.extract_results_tab(1, "Consultas Generales", current_driver))
            results.extend(self.extract_results_tab(2, "Consultas Vinculantes", current_driver))

            # Guardar resultados en cache
            self.cache.set(search_query, search_type, results)
            self._log(f"[{worker_id}] Resultados guardados en cache: '{search_query}' ({len(results)} resultados)")

        except Exception as e:
            self._error(f"[{worker_id}] Error en búsqueda '{search_query}': {e}")

        return results

    def perform_search_with_pool_driver(self, search_query, search_type="texto_libre"):
        """Realiza una búsqueda usando un driver obtenido del pool.

        Obtiene un driver disponible, navega al formulario de búsqueda,
        ejecuta la consulta y devuelve el driver al pool al finalizar
        (incluso si ocurre un error).

        Args:
            search_query (str): Texto de la consulta a buscar.
            search_type (str): Tipo de búsqueda (por defecto: 'texto_libre').

        Returns:
            list: Lista de diccionarios con los resultados. Retorna lista vacía
            si no hay drivers disponibles en el pool.
        """
        if not self.use_pool:
            return self.perform_search(search_query, search_type)

        driver = self.driver_pool.get_driver()
        if not driver:
            self._error("No hay drivers disponibles en el pool")
            return []

        try:
            # Crear WebDriverWait para este driver
            wait = WebDriverWait(driver, 10)

            # Navegar al formulario de búsqueda
            self.navigate_to_search(driver, wait)

            # Realizar búsqueda
            results = self.perform_search(search_query, search_type, driver, wait)

            return results

        finally:
            # Siempre devolver el driver al pool
            self.driver_pool.return_driver(driver)

    def perform_searches_parallel(self, search_queries, max_workers=None):
        """Realiza múltiples búsquedas en paralelo usando el pool de drivers.

        Utiliza un ``ThreadPoolExecutor`` para ejecutar las búsquedas de forma
        concurrente, distribuyendo el trabajo entre los drivers disponibles.

        Args:
            search_queries (list): Lista de tuplas ``(query, search_type)`` o
                lista de strings (en cuyo caso se usa 'texto_libre' como tipo).
            max_workers (int o None): Número máximo de hilos concurrentes.
                Por defecto usa el tamaño del pool de drivers.

        Returns:
            dict: Diccionario que mapea cada query (str) a su lista de
            resultados. Retorna diccionario vacío si use_pool=False.
        """
        if not self.use_pool:
            self._error("Búsquedas paralelas requieren usar pool de drivers (use_pool=True)")
            return {}

        # Normalizar queries a tuplas (query, search_type)
        normalized_queries = []
        for item in search_queries:
            if isinstance(item, tuple):
                normalized_queries.append(item)
            else:
                normalized_queries.append((item, "texto_libre"))

        if max_workers is None:
            max_workers = self.driver_pool.pool_size

        results = {}
        self._log(f"Iniciando {len(normalized_queries)} búsquedas en paralelo con {max_workers} workers")

        # Mostrar estado inicial del pool
        pool_status = self.driver_pool.get_pool_status()
        self._log(
            f"Estado del pool: {pool_status['available_drivers']}/{pool_status['total_drivers']} drivers disponibles")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Enviar todas las búsquedas
            future_to_query = {
                executor.submit(self.perform_search_with_pool_driver, query, search_type): query
                for query, search_type in normalized_queries
            }

            # Recoger resultados conforme van completándose
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    result = future.result()
                    results[query] = result
                    self._log(f"Completada búsqueda para: '{query}' ({len(result)} resultados)")
                except Exception as e:
                    self._error(f"Error en búsqueda paralela para '{query}': {e}")
                    results[query] = []

        # Mostrar estadísticas finales
        total_results = sum(len(r) for r in results.values())
        self._log(f"Búsquedas paralelas completadas: {len(results)} consultas, {total_results} resultados totales")

        return results

    def clear_form(self, driver=None):
        """Limpia el formulario de búsqueda haciendo clic en el botón 'Borrar'.

        Args:
            driver (webdriver.Chrome o None): Driver a utilizar.
                Si es None, usa el driver de la instancia.
        """
        current_driver = driver or self.driver
        try:
            borrar_btn = current_driver.find_element(By.XPATH, "//button[@type='reset' and text()='Borrar']")
            borrar_btn.click()
            time.sleep(1)
        except:
            pass

    def extract_results_tab(self, tab_number, tab_name, driver=None):
        """Extrae los resultados de una pestaña específica de la página de resultados.

        Navega a la pestaña indicada, recorre cada consulta listada, hace clic
        para abrir el documento completo y extrae todos los campos relevantes
        (número, órgano, fecha, normativa, descripción, cuestión, contestación).

        Args:
            tab_number (int): Número de la pestaña (1 = Consultas Generales,
                2 = Consultas Vinculantes).
            tab_name (str): Nombre descriptivo de la pestaña para los logs.
            driver (webdriver.Chrome o None): Driver a utilizar.
                Si es None, usa el driver de la instancia.

        Returns:
            list: Lista de diccionarios con los datos extraídos de cada consulta.
        """
        current_driver = driver or self.driver
        results = []

        try:
            # Hacer clic en la pestaña
            tab = current_driver.find_element(By.ID, f"ui-id-{tab_number}")
            tab.click()
            time.sleep(2)

            # Buscar contenedor de resultados
            results_div = current_driver.find_element(By.ID, f"results{tab_number}")

            # Verificar si hay mensaje de "sin resultados"
            if results_div.find_elements(By.CLASS_NAME, "message"):
                worker_id = getattr(current_driver, 'driver_id', 'main')
                self._log(f"[{worker_id}] No hay resultados en {tab_name}")
                return results

            # Buscar tabla de resultados
            table = results_div.find_element(By.CLASS_NAME, "results")
            consultas = table.find_elements(By.TAG_NAME, "td")

            for consulta in consultas:
                try:
                    # Hacer clic en la consulta
                    consulta.click()
                    time.sleep(3)  # Esperar a que cargue el documento completo

                    def get_content(item_name: str):
                        try:
                            content = doc_table.find_element(By.CLASS_NAME, item_name).text.strip()
                            return content if "\n" not in content else " ".join(content.split("\n")[1:])
                        except Exception as e:
                            print(f'Error in get_content; item_name: {item_name}.  Error: {e}')
                            return ""

                    try:
                        doc_table = current_driver.find_element(By.CLASS_NAME, "document")
                        _ = get_content("NUM-CONSULTA")
                    except:
                        doc_table = current_driver.find_element(By.CSS_SELECTOR, "table.document.doc_2")

                    # Extraer información del documento
                    num_consulta = get_content("NUM-CONSULTA")
                    organo = get_content("ORGANO")
                    fecha_salida = get_content("FECHA-SALIDA")
                    normativa = get_content("NORMATIVA")
                    descripcion = get_content("DESCRIPCION-HECHOS")
                    cuestion = get_content("CUESTION-PLANTEADA")

                    # Extraer la contestación completa (lo más importante)
                    contestacion_elements = doc_table.find_elements(By.CLASS_NAME, "CONTESTACION-COMPL")
                    contestacion = " ".join([elem.text.strip() for elem in contestacion_elements])

                    results.append({
                        'tipo': tab_name,
                        'numero_consulta': num_consulta,
                        'organo': organo,
                        'fecha_salida': fecha_salida,
                        'normativa': normativa,
                        'descripcion_hechos': descripcion,
                        'cuestion_planteada': cuestion,
                        'contestacion_completa': contestacion,
                        'relevancia': self.calculate_relevance(contestacion),
                        'instance_id': self.instance_id,  # Identificar qué instancia extrajo esto
                        'worker_id': getattr(current_driver, 'driver_id', 'main')  # Identificar qué worker
                    })

                except Exception as e:
                    worker_id = getattr(current_driver, 'driver_id', 'main')
                    self._error(f"[{worker_id}] Error extrayendo consulta individual: {e}")
                    continue

        except Exception as e:
            worker_id = getattr(current_driver, 'driver_id', 'main')
            self._error(f"[{worker_id}] Error extrayendo resultados pestaña {tab_number}: {e}")

        return results

    def calculate_relevance(self, contenido):
        """Calcula una puntuación de relevancia del contenido basada en palabras clave fiscales.

        Cuenta las ocurrencias de términos como 'deducible', 'IVA', 'IRPF',
        'exención', entre otros, para asignar una puntuación numérica.

        Args:
            contenido (str): Texto del contenido de la contestación a evaluar.

        Returns:
            int: Puntuación de relevancia (número de palabras clave encontradas).
        """
        keywords_relevantes = [
            'deducible', 'gasto', 'IVA', 'IRPF', 'actividad empresarial',
            'consultoría', 'servicios profesionales', 'informática',
            'deducción', 'beneficio fiscal', 'exención'
        ]

        contenido_lower = contenido.lower()
        score = sum(1 for keyword in keywords_relevantes if keyword in contenido_lower)
        return score

    def search_comprehensive(self, factura_data):
        """Realiza búsquedas exhaustivas en la AEAT basadas en los datos de una factura.

        Genera automáticamente las consultas de búsqueda a partir de los conceptos,
        códigos CNAE y términos fiscales de la factura, y las ejecuta en paralelo
        (si use_pool=True) o secuencialmente.

        Args:
            factura_data (dict): Datos de la factura con campos como 'conceptos',
                'receptor', 'emisor' y 'fiscal'.

        Returns:
            list: Lista consolidada de todos los resultados de las búsquedas.
        """
        keywords, fisc_terms = generar_keywords_y_fiscal_terms(factura_data)
        search_terms = self.extract_search_terms(factura_data, keywords)
        all_results = []

        self._log("Iniciando búsquedas comprehensivas...")

        # Preparar todas las consultas
        search_queries = []

        # Búsquedas por conceptos
        for concepto in search_terms['conceptos']:
            search_queries.append((concepto, "texto_libre"))

        # Búsquedas por códigos CNAE
        for cnae in search_terms['cnae_codes']:
            cnae_num = cnae.split(' - ')[0] if ' - ' in cnae else cnae
            search_queries.append((cnae_num, "texto_libre"))

        for term in fisc_terms:
            search_queries.append((term, "texto_libre"))

        if self.use_pool:
            # Ejecutar búsquedas en paralelo
            self._log(f"Ejecutando {len(search_queries)} búsquedas en paralelo...")
            parallel_results = self.perform_searches_parallel(search_queries)

            # Consolidar resultados
            for query, results in parallel_results.items():
                all_results.extend(results)

        else:
            # Ejecutar búsquedas secuencialmente (modo original)
            # scraper = AEATConsultaScraper(
            #     headless=True,
            #     verbose=True,
            #     use_pool=False,
            #     pool_size=5  # Pool más grande para este ejemplo
            # )
            # self = scraper

            self.navigate_to_search()

            # query, search_type = search_queries[0]
            # query = 'Liquidación IVA'
            for query, search_type in search_queries:
                self._log(f"Buscando: {query}")
                results = self.perform_search(query, search_type)
                all_results.extend(results)
                time.sleep(2)

        return all_results

    def get_unique_filename(self, base_name):
        """Genera un nombre de archivo único incorporando la marca de tiempo y el ID de instancia.

        Args:
            base_name (str): Nombre base del archivo (sin extensión).

        Returns:
            str: Nombre de archivo con formato '{base_name}_{timestamp}_{instance_id}'.
        """
        return f"{base_name}_{self.session_timestamp}_{self.instance_id}"

    def save_results(self, results, base_filename):
        """Guarda los resultados en formato Excel y JSON con nombres de archivo únicos.

        Ordena los resultados por relevancia descendente, elimina duplicados
        por número de consulta y exporta a ambos formatos.

        Args:
            results (list): Lista de diccionarios con los resultados a guardar.
            base_filename (str): Nombre base para los archivos de salida.

        Returns:
            pandas.DataFrame o None: DataFrame con los resultados procesados,
            o None si la lista de resultados está vacía.
        """
        if not results:
            self._log("No hay resultados para guardar")
            return

        # Crear DataFrame
        df = pd.DataFrame(results)

        # Ordenar por relevancia
        df = df.sort_values('relevancia', ascending=False)

        # Eliminar duplicados
        df = df.drop_duplicates(subset=['numero_consulta'])

        # Generar nombres únicos
        unique_filename = self.get_unique_filename(base_filename)

        # Guardar Excel
        excel_filename = f"{unique_filename}.xlsx"
        df.to_excel(excel_filename, index=False)
        self._log(f"Resultados guardados en: {excel_filename}")

        # Guardar JSON
        json_filename = f"{unique_filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self._log(f"Resultados JSON guardados en: {json_filename}")

        return df

    def analyze_benefits(self, results):
        """Analiza los resultados para identificar beneficios fiscales aplicables.

        Clasifica cada resultado según la presencia de términos relacionados con
        deducciones, exenciones y optimizaciones fiscales en la contestación.

        Args:
            results (list): Lista de diccionarios con los resultados de las búsquedas.

        Returns:
            dict: Diccionario con las claves 'deducciones_identificadas',
            'exenciones_aplicables', 'optimizaciones_fiscales',
            'consultas_relevantes' (score >= 3), 'instance_id' y 'timestamp'.
        """
        beneficios = {
            'deducciones_identificadas': [],
            'exenciones_aplicables': [],
            'optimizaciones_fiscales': [],
            'consultas_relevantes': [],
            'instance_id': self.instance_id,
            'timestamp': self.session_timestamp
        }

        for result in results:
            contenido = result['contestacion_completa'].lower()
            num_consulta = result['numero_consulta']
            cuestion = result['cuestion_planteada']
            titulo = f"{num_consulta} - {cuestion}"

            # Identificar deducciones
            if any(word in contenido for word in ['deducible', 'deducción', 'gasto deducible']):
                beneficios['deducciones_identificadas'].append({
                    'consulta': titulo,
                    'tipo': 'Deducción',
                    'relevancia': result['relevancia']
                })

            # Identificar exenciones
            if any(word in contenido for word in ['exento', 'exención', 'no sujeto']):
                beneficios['exenciones_aplicables'].append({
                    'consulta': titulo,
                    'tipo': 'Exención',
                    'relevancia': result['relevancia']
                })

            # Optimizaciones fiscales
            if any(word in contenido for word in ['optimización', 'ahorro fiscal', 'beneficio']):
                beneficios['optimizaciones_fiscales'].append({
                    'consulta': titulo,
                    'tipo': 'Optimización',
                    'relevancia': result['relevancia']
                })

            # Consultas muy relevantes (score >= 3)
            if result['relevancia'] >= 3:
                beneficios['consultas_relevantes'].append(result)

        return beneficios

    def get_cache_stats(self):
        """Obtiene las estadísticas del caché delegando a la instancia de AEATSearchCache.

        Returns:
            dict: Estadísticas del caché (ver ``AEATSearchCache.get_cache_stats``).
        """
        return self.cache.get_cache_stats()

    def cleanup_cache(self):
        """Limpia las entradas expiradas del caché e informa del resultado.

        Returns:
            int: Número de entradas eliminadas.
        """
        deleted_count = self.cache.cleanup_expired()
        self._log(f"Cache limpiado: {deleted_count} entradas expiradas eliminadas")
        return deleted_count

    def clear_cache(self):
        """Elimina completamente todo el contenido del caché (base de datos y memoria)."""
        self.cache.clear_all()
        self._log("Cache completamente limpiado")

    def cleanup_temp_files(self):
        """Elimina el directorio temporal y su contenido para esta instancia (solo modo individual)."""
        if not self.use_pool and hasattr(self, 'temp_dir'):
            try:
                import shutil
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                    self._log("Archivos temporales limpiados")
            except Exception as e:
                self._error(f"Error limpiando archivos temporales: {e}")

    def close(self):
        """Cierra el driver (o pool de drivers) y libera todos los recursos asociados."""
        if self.use_pool:
            # Cerrar pool de drivers
            if self.driver_pool:
                self.driver_pool.close_all()
                self._log("Pool de drivers cerrado")
        else:
            # Cerrar driver individual
            if self.driver:
                try:
                    self.driver.quit()
                    self._log("Driver cerrado correctamente")
                except Exception as e:
                    self._error(f"Error cerrando driver: {e}")

            # Limpiar archivos temporales
            self.cleanup_temp_files()


def main():
    """Función principal que ejecuta un flujo completo de análisis fiscal automatizado.

    Crea un scraper con pool de drivers, muestra estadísticas del caché, ejecuta
    búsquedas exhaustivas con datos de factura de ejemplo, guarda los resultados
    y presenta un análisis de beneficios fiscales.
    """
    # CONFIGURACIÓN: Cambiar estos parámetros según necesidades

    # Modo individual (original)
    # scraper = AEATConsultaScraper(headless=False, verbose=True, use_pool=False)

    # Modo pool paralelo (recomendado para múltiples búsquedas)
    scraper = AEATConsultaScraper(
        headless=True,
        verbose=True,
        use_pool=True,  # Habilitar pool de drivers
        pool_size=6  # Número de drivers en paralelo
    )

    try:
        # Mostrar estadísticas iniciales del cache
        cache_stats = scraper.get_cache_stats()
        scraper._log(f"=== ESTADÍSTICAS INICIALES DEL CACHE ===")
        scraper._log(f"Entradas totales: {cache_stats['total_entries']}")
        scraper._log(f"Hits totales: {cache_stats['total_hits']}")
        scraper._log(f"Entradas en memoria: {cache_stats['memory_cache_size']}")
        scraper._log(f"Entradas expiradas: {cache_stats['expired_entries']}")

        # Limpiar cache expirado
        if cache_stats['expired_entries'] > 0:
            scraper.cleanup_cache()

        # Mostrar estado del pool si se está usando
        if scraper.use_pool:
            pool_status = scraper.driver_pool.get_pool_status()
            scraper._log(f"=== ESTADO DEL POOL DE DRIVERS ===")
            scraper._log(f"Total drivers: {pool_status['total_drivers']}")
            scraper._log(f"Drivers disponibles: {pool_status['available_drivers']}")
            scraper._log(f"Drivers ocupados: {pool_status['busy_drivers']}")

        # Cargar datos de factura (puedes cambiar esta estructura por tu JSON)
        factura_data = {
            "conceptos": [
                {
                    "descripcion": "Servicios consultoría informática desarrollo ERP",
                    "tipo_iva": 21
                }
            ],
            "receptor": {
                "actividad_cnae": "6201 - Programación informática"
            },
            "emisor": {
                "actividad_cnae": "6202 - Consultoría informática"
            },
            "fiscal": {
                "regimen_iva": "General",
                "retencion_aplicada": True,
                "tipo_retencion": 15
            }
        }

        # Realizar búsquedas
        scraper._log("Iniciando análisis fiscal automatizado...")
        start_time = time.time()

        results = scraper.search_comprehensive(factura_data)

        end_time = time.time()
        execution_time = end_time - start_time
        scraper._log(f"Tiempo total de ejecución: {execution_time:.2f} segundos")

        if results:
            scraper._log(f"\nSe encontraron {len(results)} consultas relevantes")

            # Guardar resultados con nombres únicos
            df = scraper.save_results(results, "consultas_aeat")

            # Analizar beneficios
            beneficios = scraper.analyze_benefits(results)

            scraper._log("\n=== ANÁLISIS DE BENEFICIOS FISCALES ===")
            scraper._log(f"Deducciones identificadas: {len(beneficios['deducciones_identificadas'])}")
            scraper._log(f"Exenciones aplicables: {len(beneficios['exenciones_aplicables'])}")
            scraper._log(f"Optimizaciones fiscales: {len(beneficios['optimizaciones_fiscales'])}")
            scraper._log(f"Consultas más relevantes: {len(beneficios['consultas_relevantes'])}")

            # Mostrar top 5 más relevantes
            top_results = sorted(results, key=lambda x: x['relevancia'], reverse=True)[:5]
            scraper._log("\n=== TOP 5 CONSULTAS MÁS RELEVANTES ===")
            for i, result in enumerate(top_results, 1):
                scraper._log(
                    f"{i}. {result['numero_consulta']} - {result['cuestion_planteada']} (Relevancia: {result['relevancia']})")
                scraper._log(f"   Tipo: {result['tipo']}")
                scraper._log(f"   Resumen: {result['contestacion_completa'][:150]}...")
                scraper._log("")

            # Guardar análisis de beneficios con nombre único
            beneficios_filename = scraper.get_unique_filename("beneficios_fiscales")
            with open(f"{beneficios_filename}.json", 'w', encoding='utf-8') as f:
                json.dump(beneficios, f, ensure_ascii=False, indent=2)
            scraper._log(f"Análisis de beneficios guardado en: {beneficios_filename}.json")

        else:
            scraper._log("No se encontraron resultados relevantes")

        # Mostrar estadísticas finales del cache
        final_cache_stats = scraper.get_cache_stats()
        scraper._log(f"\n=== ESTADÍSTICAS FINALES DEL CACHE ===")
        scraper._log(f"Entradas totales: {final_cache_stats['total_entries']}")
        scraper._log(f"Hits totales: {final_cache_stats['total_hits']}")
        scraper._log(f"Promedio hits por entrada: {final_cache_stats['avg_hits_per_entry']}")
        scraper._log(f"Entradas en memoria: {final_cache_stats['memory_cache_size']}")

        # Mostrar estadísticas del pool si se está usando
        if scraper.use_pool:
            final_pool_status = scraper.driver_pool.get_pool_status()
            scraper._log(f"\n=== ESTADO FINAL DEL POOL ===")
            scraper._log(
                f"Drivers disponibles: {final_pool_status['available_drivers']}/{final_pool_status['total_drivers']}")

    except Exception as e:
        scraper._error(f"Error en ejecución principal: {e}")
    finally:
        scraper.close()


# Ejemplo de uso avanzado con búsquedas personalizadas en paralelo
def ejemplo_busquedas_paralelas():
    """Ejemplo de uso del sistema de búsquedas paralelas con consultas personalizadas.

    Demuestra cómo crear un scraper con pool de drivers y ejecutar múltiples
    consultas fiscales en paralelo, mostrando la velocidad y los resultados
    obtenidos.
    """

    scraper = AEATConsultaScraper(
        headless=True,
        verbose=True,
        use_pool=True,
        pool_size=4  # Pool más grande para este ejemplo
    )

    try:
        # Lista de consultas personalizadas
        consultas_personalizadas = [
            "deducción gastos oficina casa",
            "IVA deducible servicios profesionales",
            "IRPF retención consultoría",
            "6201",  # Código CNAE
            "6202",  # Código CNAE
            "exención IVA servicios médicos",
            "deducción gastos formación empresarial",
            "régimen especial startup",
        ]

        print("=== EJEMPLO DE BÚSQUEDAS PARALELAS PERSONALIZADAS ===")
        start_time = time.time()

        # Ejecutar todas las búsquedas en paralelo
        # self = scraper
        resultados = scraper.perform_searches_parallel(consultas_personalizadas)

        end_time = time.time()

        # Mostrar resultados
        total_resultados = 0
        for consulta, results in resultados.items():
            print(f"'{consulta}': {len(results)} resultados")
            total_resultados += len(results)

        print(f"\nTotal: {total_resultados} resultados en {end_time - start_time:.2f} segundos")
        print(f"Velocidad: ~{len(consultas_personalizadas) / (end_time - start_time):.1f} consultas/segundo")

    finally:
        scraper.close()


if __name__ == "__main__":
    # Ejecutar ejemplo principal
    main()

    # Descomentar para ejecutar ejemplo de búsquedas paralelas personalizadas
    # print("\n" + "="*60 + "\n")
    # ejemplo_busquedas_paralelas()