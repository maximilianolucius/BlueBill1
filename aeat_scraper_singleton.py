#!/usr/bin/env python3
"""
Módulo singleton para el scraper de la AEAT con pool compartido entre endpoints.

Proporciona una implementación thread-safe del patrón Singleton para
``AEATConsultaScraper``, permitiendo que múltiples endpoints de una aplicación
FastAPI compartan un único pool de drivers de Selenium. Incluye gestión
automática del ciclo de vida (inicio, reinicio por errores o tiempo, apagado),
estadísticas de uso y un wrapper asíncrono para integración con FastAPI.

Componentes principales:
    - ScraperConfig: Configuración del singleton (dataclass).
    - AEATScraperSingleton: Singleton thread-safe con auto-recovery y métricas.
    - Funciones de conveniencia: ``get_aeat_scraper``, ``initialize_aeat_scraper``,
      ``shutdown_aeat_scraper``, ``async_search_comprehensive``,
      ``aeat_scraper_lifespan``.
"""

import threading
import logging
import time
import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

# Importar tu scraper existente
try:
    from aeat_scraper import AEATConsultaScraper
except ImportError:
    # Placeholder para desarrollo
    class AEATConsultaScraper:
        def __init__(self, **kwargs):
            self.config = kwargs

        def search_comprehensive(self, data):
            return {"mock": "data"}

        def close(self):
            pass

logger = logging.getLogger(__name__)


@dataclass
class ScraperConfig:
    """Configuración del scraper singleton.

    Attributes:
        headless (bool): Ejecutar el navegador sin interfaz gráfica.
        verbose (bool): Habilitar logs detallados del scraper.
        use_pool (bool): Utilizar pool de drivers para ejecución concurrente.
        pool_size (int): Número de drivers en el pool.
        max_retries (int): Número máximo de reintentos por búsqueda fallida.
        timeout (int): Tiempo máximo de espera en segundos para operaciones web.
        auto_restart_interval (int): Intervalo en segundos para reinicio automático del pool.
    """
    headless: bool = True
    verbose: bool = True
    use_pool: bool = True
    pool_size: int = 32
    max_retries: int = 3
    timeout: int = 30
    auto_restart_interval: int = 3600  # Reiniciar pool cada hora


class AEATScraperSingleton:
    """Singleton thread-safe para AEATConsultaScraper con pool compartido.

    Garantiza que solo exista una única instancia del scraper en toda la
    aplicación, gestionando automáticamente la inicialización, el reinicio
    por errores o expiración temporal, y la recolección de estadísticas.

    Utiliza double-checked locking para la creación de la instancia y un
    ``threading.RLock`` para proteger el acceso concurrente a los contadores
    y al scraper subyacente.

    Attributes:
        config (ScraperConfig): Configuración activa del singleton.
        scraper (AEATConsultaScraper o None): Instancia del scraper subyacente.
        is_active (bool): Indica si el pool está activo y operativo.
        request_count (int): Contador total de peticiones procesadas.
        error_count (int): Contador total de errores acumulados.
        last_restart (datetime o None): Fecha/hora del último reinicio del pool.
        request_lock (threading.RLock): Lock reentrante para secciones críticas.
        stats (dict): Diccionario con estadísticas de uso acumuladas.
    """

    _instance: Optional['AEATScraperSingleton'] = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, config: Optional[ScraperConfig] = None):
        """Crea o retorna la instancia única del singleton usando double-checked locking.

        Args:
            config (ScraperConfig o None): Configuración (ignorada si la instancia ya existe).

        Returns:
            AEATScraperSingleton: La instancia única del singleton.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AEATScraperSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[ScraperConfig] = None):
        """Inicializa el singleton con la configuración proporcionada.

        Solo se ejecuta la primera vez; llamadas posteriores se ignoran gracias
        al flag ``_initialized``.

        Args:
            config (ScraperConfig o None): Configuración del scraper.
                Si es None, se usa la configuración por defecto de ScraperConfig.
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.config = config or ScraperConfig()
            self.scraper: Optional[AEATConsultaScraper] = None
            self.is_active = False
            self.request_count = 0
            self.error_count = 0
            self.last_restart = None
            self.request_lock = threading.RLock()  # Permite re-entrada
            self.stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_response_time': 0.0,
                'pool_restarts': 0,
                'created_at': datetime.now().isoformat()
            }

            self._initialized = True
            logger.info("🎯 AEATScraperSingleton initialized")

    def initialize_scraper(self) -> bool:
        """Inicializa (o reinicializa) el scraper con su pool de drivers.

        Si ya existe un scraper activo, lo cierra antes de crear uno nuevo.

        Returns:
            bool: True si la inicialización fue exitosa, False en caso de error.
        """
        try:
            with self.request_lock:
                if self.scraper is not None:
                    self.close_scraper()

                logger.info(f"🚀 Initializing AEAT Scraper pool (size: {self.config.pool_size})")

                self.scraper = AEATConsultaScraper(
                    headless=self.config.headless,
                    verbose=self.config.verbose,
                    use_pool=self.config.use_pool,
                    pool_size=self.config.pool_size
                )

                self.is_active = True
                self.last_restart = datetime.now()
                self.stats['pool_restarts'] += 1

                logger.info(f"✅ AEAT Scraper pool initialized successfully")
                return True

        except Exception as e:
            logger.error(f"❌ Error initializing AEAT Scraper: {e}")
            self.is_active = False
            return False

    def close_scraper(self):
        """Cierra el scraper subyacente y libera los recursos del pool."""
        try:
            if self.scraper is not None:
                self.scraper.close()
                self.scraper = None
                self.is_active = False
                logger.info("🔴 AEAT Scraper pool closed")
        except Exception as e:
            logger.error(f"Error closing scraper: {e}")

    def should_restart_pool(self) -> bool:
        """Determina si el pool necesita reiniciarse por inactividad, tiempo o tasa de errores.

        El pool se reinicia si: no está activo, ha excedido el intervalo de
        reinicio automático, o la tasa de errores supera el 10% (con más de
        100 peticiones procesadas).

        Returns:
            bool: True si el pool debe reiniciarse, False en caso contrario.
        """
        if not self.is_active or self.scraper is None:
            return True

        # Reiniciar si han pasado muchas horas
        if self.last_restart:
            hours_since_restart = (datetime.now() - self.last_restart).total_seconds() / 3600
            if hours_since_restart > (self.config.auto_restart_interval / 3600):
                logger.info(f"🔄 Auto-restarting pool after {hours_since_restart:.1f} hours")
                return True

        # Reiniciar si hay muchos errores
        if self.request_count > 100 and (self.error_count / self.request_count) > 0.1:
            logger.warning(f"🔄 Restarting pool due to high error rate: {self.error_count}/{self.request_count}")
            return True

        return False

    def search_comprehensive(self, factura_data: Dict[str, Any],
                             retries: int = None) -> Dict[str, Any]:
        """Ejecuta una búsqueda exhaustiva thread-safe con auto-recovery.

        Minimiza la sección crítica: solo bloquea el lock para verificar el
        estado del pool y actualizar contadores, liberándolo durante la
        llamada pesada al scraper. Implementa reintentos con backoff exponencial
        y reinicio automático del pool en el último intento.

        Args:
            factura_data (Dict[str, Any]): Datos de la factura para la búsqueda.
            retries (int o None): Número de reintentos. Si es None, usa
                ``config.max_retries``.

        Returns:
            Dict[str, Any]: Resultados de la búsqueda exhaustiva.

        Raises:
            Exception: Si falla la inicialización del pool o se agotan todos
            los reintentos.
        """
        if retries is None:
            retries = self.config.max_retries

        start_time = time.time()

        # Verificar (rápidamente) si necesitamos iniciar/reiniciar el pool
        need_restart = False
        with self.request_lock:
            if self.should_restart_pool():
                need_restart = True

        if need_restart:
            if not self.initialize_scraper():
                raise Exception("Failed to initialize AEAT Scraper pool")

        # Bucle de reintentos con lock solo para contadores y acceso a self.scraper
        for attempt in range(retries + 1):
            try:
                # Capturar referencia estable del scraper y actualizar contadores
                with self.request_lock:
                    if not self.is_active or self.scraper is None:
                        raise Exception("AEAT Scraper pool is not available")
                    self.request_count += 1
                    self.stats['total_requests'] += 1
                    scraper_ref = self.scraper

                logger.debug(f"🔍 AEAT Search attempt {attempt + 1}/{retries + 1}")

                # Llamada pesada fuera del lock
                result = scraper_ref.search_comprehensive(factura_data)

                # Actualizar estadísticas de éxito
                response_time = time.time() - start_time
                with self.request_lock:
                    self.stats['successful_requests'] += 1
                    self._update_avg_response_time(response_time)

                logger.debug(f"✅ AEAT Search completed in {response_time:.2f}s")
                return result

            except Exception as e:
                with self.request_lock:
                    self.error_count += 1
                    self.stats['failed_requests'] += 1

                logger.warning(f"⚠️ AEAT Search attempt {attempt + 1} failed: {e}")

                if attempt < retries:
                    # Reintentar con delay exponencial
                    delay = 2 ** attempt
                    logger.info(f"🔄 Retrying in {delay}s...")
                    time.sleep(delay)

                    # Para el último intento, reiniciar el pool antes de reintentar
                    if attempt == retries - 1:
                        logger.info("🔄 Restarting pool for final attempt")
                        self.initialize_scraper()
                else:
                    logger.error("❌ All AEAT Search attempts failed")
                    raise e

    def _update_avg_response_time(self, response_time: float):
        """Actualiza el tiempo promedio de respuesta usando un promedio móvil acumulativo.

        Args:
            response_time (float): Tiempo de respuesta de la última petición en segundos.
        """
        current_avg = self.stats['avg_response_time']
        successful_requests = self.stats['successful_requests']

        if successful_requests == 1:
            self.stats['avg_response_time'] = response_time
        else:
            # Promedio móvil
            self.stats['avg_response_time'] = (
                    (current_avg * (successful_requests - 1) + response_time) / successful_requests
            )

    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del singleton incluyendo métricas de rendimiento.

        Returns:
            Dict[str, Any]: Diccionario con estadísticas acumuladas, estado actual,
            tasa de errores, tiempo de actividad y peticiones por hora.
        """
        with self.request_lock:
            uptime_seconds = (datetime.now() - datetime.fromisoformat(self.stats['created_at'])).total_seconds()

            return {
                **self.stats,
                'is_active': self.is_active,
                'pool_size': self.config.pool_size,
                'current_request_count': self.request_count,
                'current_error_count': self.error_count,
                'error_rate': (self.error_count / max(self.request_count, 1)) * 100,
                'uptime_hours': uptime_seconds / 3600,
                'last_restart': self.last_restart.isoformat() if self.last_restart else None,
                'requests_per_hour': (self.stats['total_requests'] / max(uptime_seconds / 3600, 0.01))
            }

    def health_check(self) -> Dict[str, Any]:
        """Realiza una comprobación de salud del scraper y su pool.

        Returns:
            Dict[str, Any]: Diccionario con el estado ('healthy', 'unhealthy'
            o 'error'), información del pool y estadísticas completas.
        """
        try:
            with self.request_lock:
                status = "healthy" if self.is_active and self.scraper is not None else "unhealthy"

                return {
                    'status': status,
                    'pool_active': self.is_active,
                    'scraper_initialized': self.scraper is not None,
                    'pool_size': self.config.pool_size,
                    'timestamp': datetime.now().isoformat(),
                    'stats': self.get_stats()
                }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def restart_pool(self) -> bool:
        """Reinicia manualmente el pool de drivers.

        Returns:
            bool: True si el reinicio fue exitoso, False en caso de error.
        """
        logger.info("🔄 Manual pool restart requested")
        return self.initialize_scraper()

    def __del__(self):
        """Libera los recursos del scraper al destruir la instancia."""
        self.close_scraper()


# ------------------------------------------------------------------------------
# Gestión Global del Singleton
# ------------------------------------------------------------------------------

# Instancia global del singleton
_scraper_singleton: Optional[AEATScraperSingleton] = None


def get_aeat_scraper(config: Optional[ScraperConfig] = None) -> AEATScraperSingleton:
    """Obtiene la instancia singleton del scraper AEAT, creándola si no existe.

    Si se proporciona una configuración en la primera llamada, esta se usa para
    inicializar el singleton. En llamadas posteriores, si se proporciona una
    configuración diferente, se actualiza la configuración interna; el pool se
    reinicializará en la siguiente llamada a ``initialize_scraper``.

    Si no se proporciona configuración, se crea una por defecto con un tamaño
    de pool determinado por la variable de entorno ``AEAT_POOL_SIZE``
    (por defecto: 48, rango permitido: 4-64).

    Args:
        config (ScraperConfig o None): Configuración del scraper. Si es None
            y es la primera llamada, se usa la configuración por defecto.

    Returns:
        AEATScraperSingleton: La instancia única del singleton.
    """
    global _scraper_singleton
    if _scraper_singleton is None:
        if config is None:
            # default bumped with guard via env var
            try:
                import os
                requested = int(os.getenv('AEAT_POOL_SIZE', '48'))
            except Exception:
                requested = 48
            safe_pool = max(4, min(requested, 64))
            config = ScraperConfig(
                headless=True,
                verbose=True,
                use_pool=True,
                pool_size=safe_pool,
                max_retries=3,
                timeout=30,
                auto_restart_interval=3600
            )
        _scraper_singleton = AEATScraperSingleton(config)
    else:
        if config is not None:
            with _scraper_singleton.request_lock:
                _scraper_singleton.config = config
    return _scraper_singleton


def initialize_aeat_scraper(config: Optional[ScraperConfig] = None) -> bool:
    """Inicializa el scraper singleton, típicamente durante el arranque de la aplicación.

    Obtiene (o crea) la instancia singleton y arranca su pool de drivers.

    Args:
        config (ScraperConfig o None): Configuración opcional para el singleton.

    Returns:
        bool: True si la inicialización fue exitosa, False en caso de error.
    """
    try:
        scraper = get_aeat_scraper(config)
        return scraper.initialize_scraper()
    except Exception as e:
        logger.error(f"Failed to initialize AEAT scraper singleton: {e}")
        return False


def shutdown_aeat_scraper():
    """Cierra el scraper singleton y libera la referencia global.

    Debe llamarse durante el apagado de la aplicación para liberar los
    drivers de Chrome y los recursos temporales asociados.
    """
    global _scraper_singleton
    if _scraper_singleton is not None:
        _scraper_singleton.close_scraper()
        _scraper_singleton = None
        logger.info("🔴 AEAT Scraper singleton shutdown complete")


# ------------------------------------------------------------------------------
# Funciones de Conveniencia para FastAPI
# ------------------------------------------------------------------------------

async def async_search_comprehensive(factura_data: Dict[str, Any]) -> Dict[str, Any]:
    """Wrapper asíncrono de ``search_comprehensive`` para uso en endpoints FastAPI.

    Ejecuta la búsqueda en un thread pool del event loop para no bloquear
    el hilo principal de la aplicación asíncrona.

    Args:
        factura_data (Dict[str, Any]): Datos de la factura para la búsqueda.

    Returns:
        Dict[str, Any]: Resultados de la búsqueda exhaustiva.
    """
    loop = asyncio.get_event_loop()
    scraper = get_aeat_scraper()

    # Ejecutar en thread pool para no bloquear el event loop
    return await loop.run_in_executor(
        None,
        scraper.search_comprehensive,
        factura_data
    )


@asynccontextmanager
async def aeat_scraper_lifespan():
    """Context manager asíncrono para la gestión del ciclo de vida del scraper.

    Diseñado para usarse con el sistema de lifespan de FastAPI. Inicializa
    el scraper singleton al arrancar la aplicación y lo cierra limpiamente
    al apagar.

    Yields:
        None: Cede el control a la aplicación FastAPI entre el startup y el shutdown.
    """
    try:
        # Startup
        logger.info("🚀 Starting AEAT Scraper singleton...")
        success = initialize_aeat_scraper()
        if not success:
            logger.error("❌ Failed to start AEAT Scraper singleton")
        else:
            logger.info("✅ AEAT Scraper singleton started successfully")

        yield

    finally:
        # Shutdown
        logger.info("🔄 Shutting down AEAT Scraper singleton...")
        shutdown_aeat_scraper()
