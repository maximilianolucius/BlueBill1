#!/usr/bin/env python3
"""
Servidor API Unificado v4.0 - BlueBill App.

Combina dos subsistemas principales:
  1. **SmartDoc** - Gestion documental con embeddings (SentenceTransformer),
     indices FAISS para busqueda semantica y generacion de resumenes/respuestas
     mediante un backend vLLM (Qwen3-8B-AWQ).
  2. **Clasificador Fiscal AEAT** - Clasificacion automatica de facturas
     utilizando un scraper singleton contra la base de datos de la AEAT,
     enriquecido opcionalmente con la base de conocimiento SmartDoc.

Persistencia: toda la informacion (documentos, indices FAISS, metadatos de
chunks, fingerprints de duplicados, jobs de clasificacion) se almacena en
SQLite y se reconstruye en memoria al arrancar el servidor.

Dependencias externas principales:
  - FastAPI / Uvicorn (servidor HTTP y WebSocket)
  - PyPDF2 (extraccion de texto de PDF)
  - FAISS (indices de busqueda vectorial)
  - SentenceTransformer (modelo de embeddings)
  - vLLM (generacion de texto)
  - aeat_scraper_singleton (pool de navegadores para scraping AEAT)
"""

import io
import os
import json
import logging
import time
import hashlib
import re
import sqlite3
import pickle
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Iterator, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

import uuid
import asyncio
from queue import Queue
from threading import Lock
from fastapi import BackgroundTasks, WebSocket, WebSocketDisconnect

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Document processing imports
import PyPDF2
import faiss

from aeat_scraper_singleton import (
    get_aeat_scraper,
    initialize_aeat_scraper,
    shutdown_aeat_scraper,
    async_search_comprehensive,
    ScraperConfig
)

# Load environment variables
load_dotenv()

# vLLM Configuration
VLLM_BASE_URL = os.getenv('VLLM_BASE_URL', "http://172.24.250.17:8000/v1")
VLLM_MODEL = os.getenv('VLLM_MODEL', "Qwen3-8B-AWQ")
VLLM_API_KEY = os.getenv('VLLM_API_KEY', "EMPTY")

# SQLite Database Configuration
DATABASE_PATH = os.getenv('SMARTDOC_DB_PATH', 'smartdoc_persistence.db')

# Job Management Configuration
MAX_CONCURRENT_JOBS = 5
JOB_TTL_HOURS = 2400
ACTIVE_JOBS: Set[str] = set()
JOB_LOCK = Lock()
WEBSOCKET_CONNECTIONS: Dict[str, WebSocket] = {}

# Configurar el scraper singleton con tus parámetros preferidos
SMARTDOC_ENRICHMENT_ENABLED = os.getenv('SMARTDOC_ENRICHMENT', '0').strip() not in {'0', 'false', 'False'}
_requested_pool = int(os.getenv('AEAT_POOL_SIZE', '64')) # 64
_safe_pool = max(4, min(_requested_pool, 96))
SCRAPER_CONFIG = ScraperConfig(
    headless=True,
    verbose=True,
    use_pool=True,
    pool_size=_safe_pool,  # bumped with guard (max 64)
    max_retries=3,
    timeout=30,
    auto_restart_interval=3600  # Reiniciar pool cada hora
)

# AEAT cache and concurrency control
AEAT_CACHE_TTL_SECONDS = int(os.getenv('AEAT_CACHE_TTL_SECONDS', str(10 * 24 * 3600)))  # 10 días por defecto
AEAT_CACHE_MAX_ENTRIES = int(os.getenv('AEAT_CACHE_MAX_ENTRIES', '10000'))
AEAT_MAX_CONCURRENT = int(os.getenv('AEAT_MAX_CONCURRENT', str(SCRAPER_CONFIG.pool_size)))
# Guard: do not exceed pool size and reasonable upper bound
AEAT_MAX_CONCURRENT = max(1, min(AEAT_MAX_CONCURRENT, SCRAPER_CONFIG.pool_size, 64))
AEAT_SEMAPHORE = asyncio.Semaphore(AEAT_MAX_CONCURRENT)
AEAT_CACHE: Dict[str, Dict[str, Any]] = {}
AEAT_CACHE_LOCK = Lock()
AEAT_METRICS_LOCK = Lock()
AEAT_ACTIVE_REQUESTS = 0
AEAT_WAIT_COUNT = 0
AEAT_WAIT_TOTAL_SECONDS = 0.0
AEAT_WAIT_MAX_SECONDS = 0.0
AEAT_429_REJECTIONS = 0


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# SQLite Persistence Manager
# ------------------------------------------------------------------------------

class SQLitePersistenceManager:
    """Gestor de persistencia SQLite para SmartDoc y el clasificador fiscal.

    Centraliza todas las operaciones CRUD contra la base de datos SQLite:
      - Documentos (texto, chunks, embeddings, fingerprints).
      - Indices FAISS individuales y global.
      - Metadatos de chunks globales para busqueda cross-documento.
      - Jobs de clasificacion fiscal asincrona.
      - Configuraciones del sistema (contador de IDs, etc.).

    Cada metodo abre y cierra su propia conexion para evitar problemas de
    concurrencia con el modelo de hilos de FastAPI.
    """

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Inicializar el esquema SQLite creando todas las tablas necesarias si no existen."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tabla para documentos principales
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                text TEXT NOT NULL,
                chunks TEXT NOT NULL,  -- JSON serialized
                embeddings BLOB NOT NULL,  -- numpy array serialized
                fingerprint TEXT NOT NULL,
                upload_timestamp TEXT NOT NULL,
                text_length INTEGER NOT NULL,
                num_chunks INTEGER NOT NULL
            )
        ''')

        # Tabla para índices FAISS individuales
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faiss_indexes (
                doc_id INTEGER PRIMARY KEY,
                faiss_index BLOB NOT NULL,  -- FAISS index serialized
                dimension INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        ''')

        # Tabla para metadatos del índice global
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_chunk_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                global_index INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        ''')

        # Tabla para fingerprints de duplicados
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_fingerprints (
                fingerprint TEXT PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id)
            )
        ''')

        # Tabla para el índice FAISS global
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_faiss_index (
                id INTEGER PRIMARY KEY,
                faiss_index BLOB,  -- FAISS index serialized
                dimension INTEGER,
                total_vectors INTEGER
            )
        ''')

        # Tabla para configuraciones del sistema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()
        self.init_jobs_table()
        logger.info(f"✅ SQLite database initialized: {self.db_path}")


    def serialize_numpy_array(self, array: np.ndarray) -> str:
        """Serializar un array NumPy a cadena base64 (via pickle) para almacenamiento en SQLite."""
        return base64.b64encode(pickle.dumps(array)).decode('utf-8')

    def deserialize_numpy_array(self, data: str) -> np.ndarray:
        """Deserializar una cadena base64 a un array NumPy. Retorna array vacio en caso de error."""
        try:
            return pickle.loads(base64.b64decode(data.encode('utf-8')))
        except Exception as e:
            logger.error(f"Error deserializando array: {e}")
            return np.array([])  # Retornar array vacío en caso de error

    def serialize_faiss_index(self, index: faiss.IndexFlatL2) -> str:
        """Serializar un indice FAISS a cadena base64 para almacenamiento en SQLite."""
        return base64.b64encode(faiss.serialize_index(index)).decode('utf-8')

    def deserialize_faiss_index(self, data) -> faiss.IndexFlatL2:
        """Deserializar un indice FAISS soportando tanto el formato nuevo (pickle+base64) como el antiguo (bytes directos).

        Retorna None si la deserializacion falla.
        """
        try:
            if isinstance(data, str):
                # Decode base64
                decoded_bytes = base64.b64decode(data.encode('utf-8'))

                try:
                    # New format: pickle + FAISS
                    unpickled = pickle.loads(decoded_bytes)
                    return faiss.deserialize_index(unpickled)
                except:
                    # Old format: direct FAISS
                    return faiss.deserialize_index(decoded_bytes)

            elif isinstance(data, bytes):
                try:
                    # Try pickle first
                    unpickled = pickle.loads(data)
                    return faiss.deserialize_index(unpickled)
                except:
                    # Direct bytes
                    return faiss.deserialize_index(data)
            else:
                logger.error(f"Unsupported data type for FAISS: {type(data)}")
                return None

        except Exception as e:
            logger.error(f"Error deserializing FAISS index: {e}")
            return None
        


    def init_jobs_table(self):
        """Crear la tabla ``classification_jobs`` si no existe, para almacenar jobs de clasificacion fiscal."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS classification_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                progress REAL DEFAULT 0.0,
                current_step TEXT DEFAULT '',
                factura_data TEXT NOT NULL,
                result_data TEXT,
                error_message TEXT,
                estimated_duration INTEGER,
                actual_duration INTEGER
            )
        ''')

        conn.commit()
        conn.close()

    def save_job(self, job_data: Dict):
        """Guardar o actualizar un job de clasificacion en SQLite (INSERT OR REPLACE)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO classification_jobs 
                (job_id, status, created_at, started_at, completed_at, progress, 
                 current_step, factura_data, result_data, error_message, 
                 estimated_duration, actual_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job_data['job_id'], job_data['status'], job_data['created_at'],
                job_data.get('started_at'), job_data.get('completed_at'),
                job_data.get('progress', 0.0), job_data.get('current_step', ''),
                json.dumps(job_data.get('factura_data', {})),
                json.dumps(job_data.get('result_data')) if job_data.get('result_data') else None,
                job_data.get('error_message'), job_data.get('estimated_duration'),
                job_data.get('actual_duration')
            ))
            conn.commit()
        finally:
            conn.close()

    def get_job(self, job_id: str) -> Optional[Dict]:
        """Obtener un job de clasificacion por su UUID. Retorna None si no existe."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT * FROM classification_jobs WHERE job_id = ?', (job_id,))
            row = cursor.fetchone()

            if row:
                columns = [desc[0] for desc in cursor.description]
                job_data = dict(zip(columns, row))
                if job_data['factura_data']:
                    job_data['factura_data'] = json.loads(job_data['factura_data'])
                if job_data['result_data']:
                    job_data['result_data'] = json.loads(job_data['result_data'])
                return job_data
            return None
        finally:
            conn.close()

    def cleanup_expired_jobs(self):
        """Eliminar jobs cuya antiguedad supere JOB_TTL_HOURS para liberar espacio en la BD."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        expiry_time = datetime.now() - timedelta(hours=JOB_TTL_HOURS)

        try:
            cursor.execute(
                'DELETE FROM classification_jobs WHERE created_at < ?',
                (expiry_time.isoformat(),)
            )
            conn.commit()
        finally:
            conn.close()

    def save_document(self, doc_id: int, doc_data: dict):
        """Guardar un documento completo (texto, chunks, embeddings, fingerprint) en SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Serializar datos complejos
            chunks_json = json.dumps(doc_data['chunks'])
            embeddings_blob = self.serialize_numpy_array(doc_data['embeddings'])

            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (doc_id, filename, text, chunks, embeddings, fingerprint, upload_timestamp, text_length, num_chunks)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id,
                doc_data['filename'],
                doc_data['text'],
                chunks_json,
                embeddings_blob,
                doc_data['fingerprint'],
                doc_data['upload_timestamp'],
                len(doc_data['text']),
                len(doc_data['chunks'])
            ))

            conn.commit()
            logger.info(f"✅ Document {doc_id} saved to SQLite")

        except Exception as e:
            logger.error(f"❌ Error saving document {doc_id}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_faiss_index(self, doc_id: int, index: faiss.IndexFlatL2):
        """Guardar el indice FAISS individual de un documento en SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            index_blob = self.serialize_faiss_index(index)
            dimension = index.d

            cursor.execute('''
                INSERT OR REPLACE INTO faiss_indexes 
                (doc_id, faiss_index, dimension)
                VALUES (?, ?, ?)
            ''', (doc_id, index_blob, dimension))

            conn.commit()
            logger.info(f"✅ FAISS index for document {doc_id} saved")

        except Exception as e:
            logger.error(f"❌ Error saving FAISS index {doc_id}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_global_faiss_index(self, index: faiss.IndexFlatL2):
        """Guardar el indice FAISS global (unico) que abarca todos los documentos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            index_blob = self.serialize_faiss_index(index)
            dimension = index.d
            total_vectors = index.ntotal

            cursor.execute('DELETE FROM global_faiss_index')  # Solo uno global
            cursor.execute('''
                INSERT INTO global_faiss_index 
                (id, faiss_index, dimension, total_vectors)
                VALUES (1, ?, ?, ?)
            ''', (index_blob, dimension, total_vectors))

            conn.commit()
            logger.info(f"✅ Global FAISS index saved ({total_vectors} vectors)")

        except Exception as e:
            logger.error(f"❌ Error saving global FAISS index: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_global_chunk_metadata(self, metadata_list: List[Dict]):
        """Guardar la lista completa de metadatos de chunks globales (reemplaza los anteriores)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('DELETE FROM global_chunk_metadata')  # Limpiar anteriores

            for meta in metadata_list:
                cursor.execute('''
                    INSERT INTO global_chunk_metadata 
                    (doc_id, filename, chunk_index, chunk_text, global_index)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    meta['doc_id'],
                    meta['filename'],
                    meta['chunk_index'],
                    meta['chunk_text'],
                    meta['global_index']
                ))

            conn.commit()
            logger.info(f"✅ Global chunk metadata saved ({len(metadata_list)} chunks)")

        except Exception as e:
            logger.error(f"❌ Error saving global chunk metadata: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_document_fingerprints(self, fingerprints: Dict[str, int]):
        """Guardar el mapa completo de fingerprints -> doc_id para deteccion de duplicados."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('DELETE FROM document_fingerprints')  # Limpiar anteriores

            for fingerprint, doc_id in fingerprints.items():
                cursor.execute('''
                    INSERT INTO document_fingerprints (fingerprint, doc_id)
                    VALUES (?, ?)
                ''', (fingerprint, doc_id))

            conn.commit()
            logger.info(f"✅ Document fingerprints saved ({len(fingerprints)} entries)")

        except Exception as e:
            logger.error(f"❌ Error saving document fingerprints: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def save_doc_id_counter(self, counter: int):
        """Persistir el contador auto-incremental de IDs de documentos en la tabla system_config."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO system_config (key, value, updated_at)
                VALUES ('doc_id_counter', ?, ?)
            ''', (str(counter), datetime.now().isoformat()))

            conn.commit()
            logger.info(f"✅ Doc ID counter saved: {counter}")

        except Exception as e:
            logger.error(f"❌ Error saving doc ID counter: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_all_documents(self) -> Dict[int, Dict[str, Any]]:
        """Cargar todos los documentos desde SQLite y reconstruir sus embeddings en memoria."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        documents = {}

        try:
            cursor.execute('SELECT * FROM documents ORDER BY doc_id')
            rows = cursor.fetchall()

            for row in rows:
                doc_id, filename, text, chunks_json, embeddings_blob, fingerprint, upload_timestamp, text_length, num_chunks = row

                # Deserializar datos complejos
                chunks = json.loads(chunks_json)
                embeddings = self.deserialize_numpy_array(embeddings_blob)

                documents[doc_id] = {
                    'filename': filename,
                    'text': text,
                    'chunks': chunks,
                    'embeddings': embeddings,
                    'fingerprint': fingerprint,
                    'upload_timestamp': upload_timestamp
                }

            logger.info(f"✅ Loaded {len(documents)} documents from SQLite")
            return documents

        except Exception as e:
            logger.error(f"❌ Error loading documents: {e}")
            return {}
        finally:
            conn.close()

    def load_all_faiss_indexes(self) -> Dict[int, faiss.IndexFlatL2]:
        """Cargar todos los indices FAISS individuales con manejo robusto de errores.

        Los indices corruptos se omiten con un warning en lugar de abortar la carga completa.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        indexes = {}

        try:
            cursor.execute('SELECT doc_id, faiss_index FROM faiss_indexes')
            rows = cursor.fetchall()

            successful = 0
            failed = 0

            for doc_id, index_blob in rows:
                index = self.deserialize_faiss_index(index_blob)
                if index is not None:
                    indexes[doc_id] = index
                    successful += 1
                else:
                    failed += 1
                    logger.warning(f"Skipping corrupted FAISS index for doc {doc_id}")

            if successful > 0:
                logger.info(f"✅ Loaded {successful} FAISS indexes from SQLite")
            if failed > 0:
                logger.warning(f"⚠️ Failed to load {failed} FAISS indexes")

            return indexes

        except Exception as e:
            logger.error(f"❌ Error loading FAISS indexes: {e}")
            return {}
        finally:
            conn.close()

    def load_global_faiss_index(self) -> Optional[faiss.IndexFlatL2]:
        """Cargar el indice FAISS global desde SQLite, soportando multiples formatos de serializacion."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT faiss_index FROM global_faiss_index WHERE id = 1')
            row = cursor.fetchone()

            if row:
                index = self.deserialize_faiss_index(row[0])
                if index:
                    logger.info(f"✅ Loaded global FAISS index ({index.ntotal} vectors)")
                    return index
                else:
                    logger.error("❌ Failed to deserialize global FAISS index")
                    return None
            else:
                logger.info("ℹ️ No global FAISS index found in SQLite")
                return None

        except Exception as e:
            logger.error(f"❌ Error loading global FAISS index: {e}")
            return None
        finally:
            conn.close()

    def load_global_chunk_metadata(self) -> List[Dict[str, Any]]:
        """Cargar la lista ordenada de metadatos de chunks globales desde SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        metadata = []

        try:
            cursor.execute('''
                SELECT doc_id, filename, chunk_index, chunk_text, global_index 
                FROM global_chunk_metadata 
                ORDER BY global_index
            ''')
            rows = cursor.fetchall()

            for row in rows:
                doc_id, filename, chunk_index, chunk_text, global_index = row
                metadata.append({
                    'doc_id': doc_id,
                    'filename': filename,
                    'chunk_index': chunk_index,
                    'chunk_text': chunk_text,
                    'global_index': global_index
                })

            logger.info(f"✅ Loaded {len(metadata)} global chunk metadata from SQLite")
            return metadata

        except Exception as e:
            logger.error(f"❌ Error loading global chunk metadata: {e}")
            return []
        finally:
            conn.close()

    def load_document_fingerprints(self) -> Dict[str, int]:
        """Cargar el mapa fingerprint -> doc_id para restaurar la deteccion de duplicados."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        fingerprints = {}

        try:
            cursor.execute('SELECT fingerprint, doc_id FROM document_fingerprints')
            rows = cursor.fetchall()

            for fingerprint, doc_id in rows:
                fingerprints[fingerprint] = doc_id

            logger.info(f"✅ Loaded {len(fingerprints)} document fingerprints from SQLite")
            return fingerprints

        except Exception as e:
            logger.error(f"❌ Error loading document fingerprints: {e}")
            return {}
        finally:
            conn.close()

    def load_doc_id_counter(self) -> int:
        """Cargar el contador de IDs de documentos desde system_config. Retorna 1 si no existe."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT value FROM system_config WHERE key = ?', ('doc_id_counter',))
            row = cursor.fetchone()

            if row:
                counter = int(row[0])
                logger.info(f"✅ Loaded doc ID counter: {counter}")
                return counter
            else:
                logger.info("ℹ️ No doc ID counter found, starting from 1")
                return 1

        except Exception as e:
            logger.error(f"❌ Error loading doc ID counter: {e}")
            return 1
        finally:
            conn.close()

    def delete_document(self, doc_id: int):
        """Eliminar un documento y todos sus datos asociados (indice FAISS, metadata, fingerprint) de SQLite."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('DELETE FROM documents WHERE doc_id = ?', (doc_id,))
            cursor.execute('DELETE FROM faiss_indexes WHERE doc_id = ?', (doc_id,))
            cursor.execute('DELETE FROM global_chunk_metadata WHERE doc_id = ?', (doc_id,))
            cursor.execute('DELETE FROM document_fingerprints WHERE doc_id = ?', (doc_id,))

            conn.commit()
            logger.info(f"✅ Document {doc_id} deleted from SQLite")

        except Exception as e:
            logger.error(f"❌ Error deleting document {doc_id}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()


# ------------------------------------------------------------------------------
# Module Availability Tracking
# ------------------------------------------------------------------------------
AVAILABLE_MODULES = {
    "smartdoc": False,
    "fiscal_classifier": False
}

# ------------------------------------------------------------------------------
# Initialize Models and Components
# ------------------------------------------------------------------------------
print("🚀 Initializing Unified API Server v4.0 with SQLite Persistence...")
print("Loading models... (this may take a few moments)")

# Initialize SQLite Persistence Manager
persistence_manager = SQLitePersistenceManager()

try:
    # Import and initialize vLLM client
    from utils.generator_model import LangChainVLLMAdapter

    vllm_client = LangChainVLLMAdapter()
    logger.info(f"✅ vLLM client initialized: {vllm_client.base_url} - Model: {vllm_client.model_name}")

    # Force CPU usage for embedding model to avoid CUDA issues
    device = 'cpu'
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    logger.info(f"✅ Embedding model loaded: all-MiniLM-L6-v2 (device: {device})")

    AVAILABLE_MODULES["smartdoc"] = True

except Exception as e:
    logger.error(f"❌ Error loading core models: {e}")
    raise

# Try to initialize fiscal classifier components
try:
    # Import fiscal classifier components with the fixed version
    from fiscal_classifier import (
        FiscalPattern, FiscalRecommendation, FacturaDataExtractor,
        ConsultasTributariasProcessor, ConsultasRecommendationEngine,
        EnhancedFiscalClassifier
    )

    AVAILABLE_MODULES["fiscal_classifier"] = True
    logger.info("✅ Fiscal classifier modules imported successfully")

except Exception as e:
    logger.warning(f"⚠️ Fiscal classifier not available: {e}")


    # Create placeholder classes to prevent import errors
    class FiscalPattern:
        pass


    class FiscalRecommendation:
        pass


    class FacturaDataExtractor:
        pass


    class ConsultasTributariasProcessor:
        pass


    class ConsultasRecommendationEngine:
        pass


    class EnhancedFiscalClassifier:
        def __init__(self, *args, **kwargs):
            pass

        def classify_expense_with_precedents(self, *args, **kwargs):
            raise HTTPException(status_code=503, detail="Fiscal classifier not available")

        def validate_factura_input(self, *args, **kwargs):
            raise HTTPException(status_code=503, detail="Fiscal classifier not available")

# ------------------------------------------------------------------------------
# FastAPI Application
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Unified API Server v4.0",
    description="SmartDoc Document Management + Fiscal Classification AEAT con Persistencia SQLite",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Global Variables (Reconstruidas desde SQLite)
# ------------------------------------------------------------------------------

# SmartDoc Storage - Se reconstruyen desde SQLite
documents: Dict[int, Dict[str, Any]] = {}
faiss_indexes: Dict[int, faiss.IndexFlatL2] = {}
doc_id_counter = 1

# Global FAISS index for cross-document search
global_faiss_index: faiss.IndexFlatL2 = None
global_chunk_metadata: List[Dict[str, Any]] = []

# Document fingerprint tracking for duplicate detection
document_fingerprints: Dict[str, int] = {}

# Fiscal Classifier
fiscal_classifier: Optional[EnhancedFiscalClassifier] = None


# ------------------------------------------------------------------------------
# Error Handling Helpers (sanitized for end users)
# ------------------------------------------------------------------------------

def _new_correlation_id() -> str:
    """Generar un ID de correlacion unico (UUID hex) para trazabilidad de errores."""
    return uuid.uuid4().hex



def _safe_http_error(component: str,
                     stage: str,
                     error_code: str,
                     user_message: str,
                     status_code: int = 500,
                     retryable: bool = True,
                     exc: Optional[Exception] = None) -> HTTPException:
    """Crear un HTTPException sanitizado con ID de correlacion para trazabilidad.

    No incluye texto crudo de la excepcion en la respuesta al cliente para evitar
    filtrar detalles internos. Registra el error completo del lado del servidor
    con el ID de correlacion para facilitar el diagnostico.
    """
    cid = _new_correlation_id()
    if exc is not None:
        # Log full traceback for operators; correlation id ties logs to response
        logger.exception(f"[{cid}] {component}::{stage} failed ({error_code})")
    else:
        logger.error(f"[{cid}] {component}::{stage} failed ({error_code})")

    detail = {
        "message": user_message,
        "error_code": error_code,
        "component": component,
        "stage": stage,
        "retryable": retryable,
        "correlation_id": cid,
    }

    return HTTPException(status_code=status_code, detail=detail, headers={"X-Correlation-ID": cid})


def reconstruct_variables_from_sqlite():
    """Reconstruir todas las variables globales en memoria desde SQLite al arrancar el servidor."""
    global documents, faiss_indexes, doc_id_counter
    global global_faiss_index, global_chunk_metadata, document_fingerprints

    logger.info("🔄 Reconstructing variables from SQLite...")

    # Cargar documentos
    documents = persistence_manager.load_all_documents()

    # Cargar índices FAISS individuales
    faiss_indexes = persistence_manager.load_all_faiss_indexes()

    # Cargar índice FAISS global
    global_faiss_index = persistence_manager.load_global_faiss_index()

    # Cargar metadatos de chunks globales
    global_chunk_metadata = persistence_manager.load_global_chunk_metadata()

    # Cargar fingerprints de documentos
    document_fingerprints = persistence_manager.load_document_fingerprints()

    # Cargar contador de IDs
    doc_id_counter = persistence_manager.load_doc_id_counter()

    logger.info(f"📊 Reconstruction completed:")
    logger.info(f"  - Documents: {len(documents)}")
    logger.info(f"  - FAISS indexes: {len(faiss_indexes)}")
    logger.info(f"  - Global chunks: {len(global_chunk_metadata)}")
    logger.info(f"  - Fingerprints: {len(document_fingerprints)}")
    logger.info(f"  - Next doc ID: {doc_id_counter}")


def save_all_to_sqlite():
    """Persistir todas las variables globales en SQLite (usado en shutdown y backup manual)."""
    try:
        # Guardar contador de IDs
        persistence_manager.save_doc_id_counter(doc_id_counter)

        # Guardar fingerprints
        persistence_manager.save_document_fingerprints(document_fingerprints)

        # Guardar metadatos globales
        persistence_manager.save_global_chunk_metadata(global_chunk_metadata)

        # Guardar índice global si existe
        if global_faiss_index is not None:
            persistence_manager.save_global_faiss_index(global_faiss_index)

        logger.info("✅ All variables saved to SQLite")

    except Exception as e:
        logger.error(f"❌ Error saving to SQLite: {e}")


# ------------------------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------------------------

# System Models
class SystemInfo(BaseModel):
    """Modelo de respuesta para el endpoint raiz con informacion general del servicio."""

    service: str
    version: str
    description: str
    available_modules: Dict[str, bool]
    documentation: Dict[str, str]
    endpoints: Dict[str, str]
    timestamp: str
    persistence: Dict[str, Any]


class HealthResponse(BaseModel):
    """Modelo de respuesta del health check con estado de cada componente."""

    status: str
    version: str
    timestamp: str
    components: Dict[str, Any]


class CapabilitiesResponse(BaseModel):
    """Modelo de respuesta que lista modulos disponibles y sus endpoints."""

    version: str
    available_modules: List[str]
    endpoints: Dict[str, Dict[str, str]]
    timestamp: str


# SmartDoc Models
class QueryRequest(BaseModel):
    """Modelo de solicitud para consultas sobre documentos (consulta en lenguaje natural)."""

    query: str


class UploadResponse(BaseModel):
    """Modelo de respuesta tras la subida de un documento, incluyendo info de duplicados."""

    doc_id: int
    message: str
    duplicate_info: Optional[Dict[str, Any]] = None
    persistence_status: str


class SummaryResponse(BaseModel):
    """Modelo de respuesta con el resumen generado por vLLM de un documento."""

    doc_id: int
    summary: str


class QueryResponse(BaseModel):
    """Modelo de respuesta para consultas sobre un documento individual."""

    doc_id: int
    query: str
    answer: str
    context_chunks: List[str]


class GlobalQueryResponse(BaseModel):
    """Modelo de respuesta para consultas cross-documento con atribucion de fuentes."""

    query: str
    answer: str
    context_chunks: List[str]
    sources: List[Dict[str, Any]]
    total_documents_searched: int


class DocumentInfo(BaseModel):
    """Modelo con metadatos basicos de un documento almacenado."""

    doc_id: int
    filename: str
    text_length: int
    num_chunks: int
    upload_timestamp: str


class PersistenceStats(BaseModel):
    """Modelo con estadisticas de la capa de persistencia SQLite."""

    database_path: str
    total_documents: int
    total_chunks: int
    database_size_mb: float
    last_backup: str

# Fiscal Classifier Models
class FacturaRequest(BaseModel):
    """Modelo de solicitud que representa una factura para clasificacion fiscal.

    Contiene las secciones estandar de una factura: identificacion, conceptos,
    receptor, emisor, importes, contexto empresarial, datos fiscales, relacion
    comercial y fechas.
    """

    identificacion: Optional[Dict] = None
    conceptos: List[Dict]
    receptor: Optional[Dict] = None
    emisor: Optional[Dict] = None
    importes: Optional[Dict] = None
    contexto_empresarial: Optional[Dict] = None
    fiscal: Optional[Dict] = None
    relacion_comercial: Optional[Dict] = None
    fechas: Optional[Dict] = None


class FiscalClassificationResponse(BaseModel):
    """Modelo de respuesta con el resultado completo de la clasificacion fiscal.

    Incluye clasificacion AEAT, analisis de deducibilidad, oportunidades fiscales,
    alertas de cumplimiento, consultas vinculantes aplicables y, opcionalmente,
    resultados enriquecidos desde SmartDoc.
    """

    clasificacion: Dict
    deducibilidad: Dict
    oportunidades_fiscales: List[Dict]
    alertas_cumplimiento: List[Dict]
    analisis_relacion_comercial: Dict
    consultas_vinculantes_aplicables: List[str]
    confidence_score: float
    metadata: Optional[Dict] = None
    smartdoc_results: Optional[Dict] = None  # SmartDoc search results


class ValidationResponse(BaseModel):
    """Modelo de respuesta de la validacion estructural de una factura."""

    valid: bool
    message: Optional[str] = None
    error: Optional[str] = None
    conceptos_count: Optional[int] = None
    sections_present: Optional[List[str]] = None
    timestamp: str

class ClassificationJob(BaseModel):
    """Modelo que representa el estado de un job de clasificacion fiscal asincrono."""

    job_id: str
    status: str  # "pending", "processing", "completed", "failed", "cancelled"
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0  # 0-100%
    current_step: str = ""
    estimated_duration: Optional[int] = None  # seconds
    actual_duration: Optional[int] = None

class JobStatusResponse(BaseModel):
    """Modelo de respuesta para consultar el progreso de un job de clasificacion."""

    job_id: str
    status: str
    progress: float
    current_step: str
    created_at: str
    estimated_remaining: Optional[str] = None

class JobResultResponse(BaseModel):
    """Modelo de respuesta con el resultado final (o error) de un job completado."""

    job_id: str
    status: str
    result: Optional[FiscalClassificationResponse] = None
    error: Optional[str] = None
    completed_at: str
    actual_duration: int

# ------------------------------------------------------------------------------
# SmartDoc Helper Functions
# ------------------------------------------------------------------------------

def extract_search_query_from_factura(factura_dict: Dict) -> str:
    """Extraer terminos de busqueda relevantes de una factura para consultar SmartDoc.

    Combina descripciones de conceptos, actividad principal y sector empresarial
    en una sola cadena de texto (max 800 caracteres).
    """
    query_parts = []

    # Extract from conceptos
    conceptos = factura_dict.get('conceptos', [])
    for concepto in conceptos[:8]:
        if 'descripcion' in concepto:
            query_parts.append(concepto['descripcion'])
        if 'concepto' in concepto:
            query_parts.append(concepto['concepto'])

    # Extract from contexto empresarial
    contexto = factura_dict.get('contexto_empresarial', {})
    if 'actividad_principal' in contexto:
        query_parts.append(contexto['actividad_principal'])
    if 'sector' in contexto:
        query_parts.append(contexto['sector'])

    # Join and clean up
    query = " ".join(query_parts)
    return query[:800]  # Limit query length


def search_smartdoc_for_fiscal(query: str, max_results: int = 12) -> Optional[Dict]:
    """Buscar en la base documental SmartDoc informacion relevante para clasificacion fiscal.

    Realiza busqueda semantica en el indice FAISS global, genera un analisis
    fiscal con vLLM y devuelve los chunks relevantes junto con las fuentes.
    Retorna None si no hay resultados o la consulta esta vacia.
    """
    search_start = time.perf_counter()
    logger.info(
        "[SmartDoc] 🔍 Inicio búsqueda para consulta fiscal (%.120s)",
        query.strip().replace("\n", " ") if query else ""
    )
    try:
        if not query.strip() or not global_chunk_metadata:
            return None

        # Create query embedding
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)

        # Search global index
        retrieved_chunks, metadata = search_global_index(query_embedding, k=max_results)

        if not retrieved_chunks:
            logger.info("[SmartDoc] ℹ️ Sin resultados relevantes (%.2fs)", time.perf_counter() - search_start)
            return None

        system_prompt = """Estás analizando documentos para información fiscal y tributaria. 
        Enfócate en deducibilidad, implicaciones fiscales y regulaciones relevantes.
        Responde SIEMPRE en español."""

        context = "\n\n".join(retrieved_chunks[:3])
        prompt = f"""Basándote en estos documentos, proporciona información fiscal y tributaria para: {query}

        Contexto:
        {context}

        Análisis Fiscal:"""

        vllm_client.max_tokens = max(getattr(vllm_client, 'max_tokens', 0) or 0, 6000)
        vllm_client.temperature = 0.3
        llm_start = time.perf_counter()
        analysis = vllm_client.generate(prompt, system_instruction=system_prompt)
        if '</think>' in analysis:
            analysis = analysis.split('</think>')[-1]
        logger.info(
            "[SmartDoc] ✅ Generación LLM completada en %.2fs (total %.2fs)",
            time.perf_counter() - llm_start,
            time.perf_counter() - search_start
        )

        # Prepare sources info
        sources = []
        seen_docs = {}
        for meta in metadata:
            doc_key = f"{meta['doc_id']}_{meta['filename']}"
            if doc_key not in seen_docs:
                seen_docs[doc_key] = {
                    "doc_id": meta['doc_id'],
                    "filename": meta['filename'],
                    "relevance_chunks": 0
                }
            seen_docs[doc_key]["relevance_chunks"] += 1

        return {
            "query_used": query,
            "analysis": analysis,
            "relevant_chunks": retrieved_chunks,
            "sources": list(seen_docs.values()),
            "total_sources": len(seen_docs)
        }

    except Exception as e:
        logger.error(
            "[SmartDoc] ❌ Error tras %.2fs: %s",
            time.perf_counter() - search_start,
            e
        )
        return {"error": str(e)}


async def search_smartdoc_for_fiscal_async(query: str, max_results: int = 12) -> Optional[Dict]:
    """Ejecutar la busqueda fiscal SmartDoc en un hilo worker para no bloquear el event loop."""
    return await asyncio.to_thread(search_smartdoc_for_fiscal, query, max_results)


def _normalize_aeat_results(aeat_results: Any) -> Dict:
    """Normalizar la salida del scraper AEAT a una estructura dict consistente.

    Soporta entradas de tipo dict, list o formatos inesperados, siempre
    devolviendo un dict con la clave 'consultas_vinculantes'.
    """
    if isinstance(aeat_results, dict):
        return aeat_results.copy()

    if isinstance(aeat_results, list):
        return {
            'consultas_vinculantes': aeat_results,
            'metadata': {
                'fuente': 'aeat_scraper',
                'total_consultas': len(aeat_results),
                'timestamp_normalizado': datetime.now().isoformat(),
            }
        }

    logger.warning(
        "AEAT results received in unexpected format (%s); wrapping into empty structure",
        type(aeat_results).__name__,
    )
    return {
        'consultas_vinculantes': [],
        'metadata': {
            'fuente': 'aeat_scraper',
            'total_consultas': 0,
            'timestamp_normalizado': datetime.now().isoformat(),
            'detalle': 'estructura no reconocida, se devolvió vacío'
        }
    }


def merge_aeat_and_smartdoc_results(aeat_results: Any, smartdoc_results: Optional[Dict]) -> Dict:
    """Fusionar los resultados del scraper AEAT con el analisis de SmartDoc.

    Si el enriquecimiento SmartDoc esta habilitado y hay analisis disponible,
    se crea una pseudo-consulta adicional y se anaden metadatos de fusion.
    """
    merged_results = _normalize_aeat_results(aeat_results)

    if not SMARTDOC_ENRICHMENT_ENABLED:
        return merged_results

    if smartdoc_results and smartdoc_results.get('analysis'):
        # Add SmartDoc insights to existing AEAT data
        if 'consultas_adicionales' not in merged_results:
            merged_results['consultas_adicionales'] = []

        # Create pseudo-consulta from SmartDoc analysis
        smartdoc_consulta = {
            'numero': f"SMARTDOC_{datetime.now().strftime('%Y%m%d')}",
            'fecha': datetime.now().strftime('%Y-%m-%d'),
            'organo': 'SmartDoc Internal Analysis',
            'descripcion': smartdoc_results['analysis'],
            'tipo': 'internal_knowledge',
            'relevancia': 0.9,
            'fuentes_documentales': smartdoc_results.get('sources', []),
            'total_fuentes': smartdoc_results.get('total_sources', 0)
        }

        merged_results['consultas_adicionales'].append(smartdoc_consulta)

        # Enhance existing consultas with document context if available
        if 'consultas_vinculantes' in merged_results:
            for consulta in merged_results['consultas_vinculantes']:
                consulta['contexto_documental'] = f"Consultar también: {smartdoc_results['query_used']}"

        # Add metadata about the merge
        if 'metadata_fusion' not in merged_results:
            merged_results['metadata_fusion'] = {}

        merged_results['metadata_fusion'].update({
            'smartdoc_enabled': True,
            'documentos_consultados': smartdoc_results.get('total_sources', 0),
            'query_smartdoc': smartdoc_results.get('query_used', ''),
            'timestamp_fusion': datetime.now().isoformat()
        })

    return merged_results

def extract_text(file: UploadFile) -> str:
    """Extraer texto de un archivo subido (PDF via PyPDF2 o texto plano UTF-8)."""
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(file.file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            file.file.seek(0)
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PDF processing error: {str(e)}")
    else:
        try:
            content = file.file.read().decode("utf-8")
            file.file.seek(0)
            return content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Text file reading error: {str(e)}")


def create_document_fingerprint(text: str) -> str:
    """Crear un fingerprint del documento usando los primeros 128 y ultimos 128 caracteres.

    Se utiliza para detectar duplicados de forma rapida sin comparar el texto completo.
    """
    text = text.strip()
    if len(text) <= 256:
        return text
    return text[:128] + text[-128:]


def remove_document_from_global_index(doc_id: int):
    """Eliminar todos los chunks de un documento del indice FAISS global y reconstruirlo."""
    global global_faiss_index, global_chunk_metadata

    if not global_chunk_metadata:
        return

    indices_to_remove = []
    new_metadata = []

    for i, meta in enumerate(global_chunk_metadata):
        if meta['doc_id'] != doc_id:
            new_metadata.append(meta)
        else:
            indices_to_remove.append(i)

    if indices_to_remove:
        global_chunk_metadata = new_metadata

        if new_metadata:
            all_embeddings = []
            for meta in new_metadata:
                old_doc_id = meta['doc_id']
                chunk_idx = meta['chunk_index']
                if old_doc_id in documents:
                    doc_embeddings = documents[old_doc_id]['embeddings']
                    if chunk_idx < len(doc_embeddings):
                        all_embeddings.append(doc_embeddings[chunk_idx])

            if all_embeddings:
                embeddings_array = np.array(all_embeddings)
                global_faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])
                global_faiss_index.add(embeddings_array)

                for i, meta in enumerate(global_chunk_metadata):
                    meta['global_index'] = i

                # Guardar en SQLite
                persistence_manager.save_global_faiss_index(global_faiss_index)
                persistence_manager.save_global_chunk_metadata(global_chunk_metadata)
        else:
            global_faiss_index = None



# ------------------------------------------------------------------------------
# AEAT Cache + Concurrency Helpers
# ------------------------------------------------------------------------------

def _normalize_factura_for_cache(factura: Dict) -> str:
    """Generar una clave de cache SHA-256 a partir de la factura serializada de forma determinista."""
    try:
        normalized = json.dumps(factura, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        normalized = str(factura)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _aeat_cache_get(cache_key: str) -> Optional[Dict]:
    """Obtener un resultado cacheado de AEAT. Retorna None si no existe o ha expirado (TTL)."""
    now = datetime.now().timestamp()
    with AEAT_CACHE_LOCK:
        entry = AEAT_CACHE.get(cache_key)
        if not entry:
            return None
        if now - entry['ts'] > AEAT_CACHE_TTL_SECONDS:
            AEAT_CACHE.pop(cache_key, None)
            return None
        return entry['data']


def _aeat_cache_set(cache_key: str, data: Dict) -> None:
    """Almacenar un resultado AEAT en cache. Expulsa la entrada mas antigua si se alcanza el limite."""
    now = datetime.now().timestamp()
    with AEAT_CACHE_LOCK:
        if len(AEAT_CACHE) >= AEAT_CACHE_MAX_ENTRIES:
            # remove oldest
            oldest_key = min(AEAT_CACHE.items(), key=lambda kv: kv[1]['ts'])[0]
            AEAT_CACHE.pop(oldest_key, None)
        AEAT_CACHE[cache_key] = { 'ts': now, 'data': data }


async def get_aeat_results_with_cache(factura_dict: Dict, allow_wait: bool = True) -> Dict:
    """Obtener resultados AEAT con cache en memoria y control de concurrencia via semaforo.

    Flujo:
      1. Verificar cache (TTL configurable, por defecto 10 dias).
      2. Si no hay cache, adquirir semaforo (max AEAT_MAX_CONCURRENT peticiones simultaneas).
         - ``allow_wait=True``: espera indefinidamente al semaforo (usado en jobs async).
         - ``allow_wait=False``: espera 0.5s maximo; si no se obtiene, devuelve HTTP 429.
      3. Doble verificacion de cache tras la espera.
      4. Ejecutar busqueda AEAT y almacenar resultado en cache.
    """
    global AEAT_WAIT_COUNT, AEAT_WAIT_TOTAL_SECONDS, AEAT_WAIT_MAX_SECONDS, AEAT_ACTIVE_REQUESTS, AEAT_429_REJECTIONS
    cache_key = _normalize_factura_for_cache(factura_dict)
    cached = _aeat_cache_get(cache_key)
    if cached is not None:
        return cached

    # Concurrency gate
    acquired = False
    wait_started = time.perf_counter()
    try:
        if allow_wait:
            await AEAT_SEMAPHORE.acquire()
            acquired = True
            wait_time = time.perf_counter() - wait_started
            with AEAT_METRICS_LOCK:
                AEAT_WAIT_COUNT += 1
                AEAT_WAIT_TOTAL_SECONDS += wait_time
                AEAT_WAIT_MAX_SECONDS = max(AEAT_WAIT_MAX_SECONDS, wait_time)
                AEAT_ACTIVE_REQUESTS += 1
        else:
            try:
                await asyncio.wait_for(AEAT_SEMAPHORE.acquire(), timeout=0.5)
                acquired = True
                wait_time = time.perf_counter() - wait_started
                with AEAT_METRICS_LOCK:
                    AEAT_WAIT_COUNT += 1
                    AEAT_WAIT_TOTAL_SECONDS += wait_time
                    AEAT_WAIT_MAX_SECONDS = max(AEAT_WAIT_MAX_SECONDS, wait_time)
                    AEAT_ACTIVE_REQUESTS += 1
            except asyncio.TimeoutError:
                # Too many concurrent AEAT requests
                with AEAT_METRICS_LOCK:
                    AEAT_429_REJECTIONS += 1
                raise HTTPException(status_code=429, detail="Too many AEAT requests, please retry later", headers={"Retry-After": "5"})

        # Double-check cache after waiting
        cached = _aeat_cache_get(cache_key)
        if cached is not None:
            return cached

        # Perform AEAT search
        results = await async_search_comprehensive(factura_dict)
        _aeat_cache_set(cache_key, results)
        return results
    finally:
        if acquired:
            with AEAT_METRICS_LOCK:
                AEAT_ACTIVE_REQUESTS = max(0, AEAT_ACTIVE_REQUESTS - 1)
            AEAT_SEMAPHORE.release()

    indices_to_remove = []
    new_metadata = []

    for i, meta in enumerate(global_chunk_metadata):
        if meta['doc_id'] != doc_id:
            new_metadata.append(meta)
        else:
            indices_to_remove.append(i)

    if indices_to_remove:
        global_chunk_metadata = new_metadata

        if new_metadata:
            all_embeddings = []
            for meta in new_metadata:
                old_doc_id = meta['doc_id']
                chunk_idx = meta['chunk_index']
                if old_doc_id in documents:
                    doc_embeddings = documents[old_doc_id]['embeddings']
                    if chunk_idx < len(doc_embeddings):
                        all_embeddings.append(doc_embeddings[chunk_idx])

            if all_embeddings:
                embeddings_array = np.array(all_embeddings)
                global_faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])
                global_faiss_index.add(embeddings_array)

                for i, meta in enumerate(global_chunk_metadata):
                    meta['global_index'] = i

                # Guardar en SQLite
                persistence_manager.save_global_faiss_index(global_faiss_index)
                persistence_manager.save_global_chunk_metadata(global_chunk_metadata)
        else:
            global_faiss_index = None


def handle_duplicate_document(fingerprint: str, new_doc_id: int, filename: str) -> Dict[str, Any]:
    """Detectar y gestionar documentos duplicados basandose en fingerprint.

    Si ya existe un documento con el mismo fingerprint, lo reemplaza (elimina
    el anterior de memoria y SQLite) y devuelve informacion del reemplazo.
    """
    if fingerprint in document_fingerprints:
        old_doc_id = document_fingerprints[fingerprint]
        old_filename = documents[old_doc_id]['filename'] if old_doc_id in documents else "Unknown"

        if old_doc_id in documents:
            remove_document_from_global_index(old_doc_id)
            del documents[old_doc_id]

            # Eliminar de SQLite
            persistence_manager.delete_document(old_doc_id)

        if old_doc_id in faiss_indexes:
            del faiss_indexes[old_doc_id]

        document_fingerprints[fingerprint] = new_doc_id

        # Guardar fingerprints actualizados
        persistence_manager.save_document_fingerprints(document_fingerprints)

        return {
            "duplicate_detected": True,
            "action": "replaced",
            "old_doc_id": old_doc_id,
            "old_filename": old_filename,
            "new_doc_id": new_doc_id,
            "new_filename": filename
        }
    else:
        document_fingerprints[fingerprint] = new_doc_id

        # Guardar fingerprints actualizados
        persistence_manager.save_document_fingerprints(document_fingerprints)

        return {
            "duplicate_detected": False,
            "action": "added",
            "new_doc_id": new_doc_id,
            "new_filename": filename
        }


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Dividir texto en fragmentos (chunks) con solapamiento del 10% para preservar contexto."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    overlap = chunk_size // 10

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def create_embeddings(chunks: List[str]) -> np.ndarray:
    """Generar embeddings vectoriales para una lista de chunks usando SentenceTransformer."""
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    return embeddings


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Construir un indice FAISS IndexFlatL2 (busqueda exacta por distancia L2) a partir de embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def initialize_global_index(dimension: int):
    """Inicializar el indice FAISS global si aun no existe (lazy initialization)."""
    global global_faiss_index
    if global_faiss_index is None:
        global_faiss_index = faiss.IndexFlatL2(dimension)


def add_to_global_index(embeddings: np.ndarray, doc_id: int, filename: str, chunks: List[str]):
    """Anadir los embeddings de un documento al indice FAISS global y actualizar los metadatos en SQLite."""
    global global_faiss_index, global_chunk_metadata

    if global_faiss_index is None:
        initialize_global_index(embeddings.shape[1])

    global_faiss_index.add(embeddings)

    for i, chunk in enumerate(chunks):
        global_chunk_metadata.append({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "chunk_text": chunk,
            "global_index": len(global_chunk_metadata)
        })

    # Guardar en SQLite
    persistence_manager.save_global_faiss_index(global_faiss_index)
    persistence_manager.save_global_chunk_metadata(global_chunk_metadata)


def search_global_index(query_embedding: np.ndarray, k: int = 10) -> tuple:
    """Buscar en el indice FAISS global los k chunks mas similares y devolver textos y metadatos."""
    global global_faiss_index, global_chunk_metadata

    if global_faiss_index is None or len(global_chunk_metadata) == 0:
        return [], []

    distances, indices = global_faiss_index.search(query_embedding, k)

    retrieved_metadata = []
    retrieved_chunks = []

    for idx in indices[0]:
        if 0 <= idx < len(global_chunk_metadata):
            metadata = global_chunk_metadata[idx]
            retrieved_metadata.append(metadata)
            retrieved_chunks.append(metadata["chunk_text"])

    return retrieved_chunks, retrieved_metadata


def summarize_with_vllm(text: str) -> str:
    """Generar un resumen del texto usando vLLM.

    Para textos cortos (<2000 chars) genera un resumen directo.
    Para textos largos, resume por secciones y luego crea un resumen integrador.
    """
    system_prompt = """Eres un experto resumidor de documentos. Crea resúmenes concisos y precisos que capturen los puntos principales y conocimientos clave. 
    Enfócate en la información más importante mientras mantienes claridad y coherencia.
    Responde SIEMPRE en español."""

    vllm_client.max_tokens = max(getattr(vllm_client, 'max_tokens', 0) or 0, 6000)
    vllm_client.temperature = 0.3

    try:
        if len(text) < 2000:
            prompt = f"""Resume este documento en 3-4 oraciones, destacando los puntos principales:

            {text}

            Resumen:"""

            return vllm_client.generate(prompt, system_instruction=system_prompt)
        else:
            chunks = chunk_text(text, chunk_size=3000)
            chunk_summaries = []

            for i, chunk in enumerate(chunks[:8]):
                chunk_prompt = f"""Resume esta sección (parte {i + 1}) en 2-3 oraciones:

                {chunk}

                Resumen de la sección:"""

                summary = vllm_client.generate(chunk_prompt, system_instruction=system_prompt)
                if summary.strip():
                    chunk_summaries.append(summary.strip())

            if not chunk_summaries:
                raise Exception("No chunk summaries produced.")

            combined_summaries = "\n\n".join(chunk_summaries)
            final_prompt = f"""Crea un resumen final integral a partir de estos resúmenes de sección:

            {combined_summaries}

            Resumen final integral:"""

            vllm_client.max_tokens = max(getattr(vllm_client, 'max_tokens', 0) or 0, 6000)
            return vllm_client.generate(final_prompt, system_instruction=system_prompt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM summarization error: {str(e)}")


def answer_with_vllm(query: str, context_chunks: List[str]) -> str:
    """Responder preguntas usando vLLM con contexto de chunks de un unico documento."""
    system_prompt = """Eres un asistente útil que responde preguntas basándose en el contexto proporcionado. 
    Sé preciso e informativo. Si la respuesta no está claramente en el contexto, dilo de manera cortés. 
    Siempre basa tu respuesta en la información proporcionada.
    Responde SIEMPRE en español."""

    vllm_client.max_tokens = max(getattr(vllm_client, 'max_tokens', 0) or 0, 6000)
    vllm_client.temperature = 0.2

    try:
        context = "\n\n".join(context_chunks[:8])

        prompt = f"""Información de Contexto:
        {context}

        Pregunta: {query}

        Basándote en el contexto anterior, proporciona una respuesta detallada:"""

        return vllm_client.generate(prompt, system_instruction=system_prompt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM Q&A error: {str(e)}")


def answer_global_with_vllm(query: str, context_chunks: List[str], metadata: List[Dict[str, Any]]) -> str:
    """Responder preguntas usando vLLM con contexto de multiples documentos y atribucion de fuentes."""
    system_prompt = """Eres un asistente útil que responde preguntas basándose en el contexto proporcionado de múltiples documentos. 
    Sé preciso e informativo. Cuando sea posible, menciona qué documento(s) respaldan tu respuesta. 
    Si la respuesta no está claramente en el contexto, dilo de manera cortés. Siempre basa tu respuesta en la información proporcionada.
    Responde SIEMPRE en español."""

    vllm_client.max_tokens = max(getattr(vllm_client, 'max_tokens', 0) or 0, 6000)
    vllm_client.temperature = 0.2

    try:
        context_with_sources = ""
        seen_docs = set()

        for i, (chunk, meta) in enumerate(zip(context_chunks[:8], metadata[:8])):
            doc_info = f"[Document: {meta['filename']}]"
            context_with_sources += f"{doc_info}\n{chunk}\n\n"
            seen_docs.add(meta['filename'])

        prompt = f"""Información de Contexto de {len(seen_docs)} documento(s):
        {context_with_sources}

        Pregunta: {query}

        Basándote en el contexto anterior de múltiples documentos, proporciona una respuesta integral. 
        Cuando sea relevante, menciona qué documento(s) respaldan puntos específicos:"""

        return vllm_client.generate(prompt, system_instruction=system_prompt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM Global Q&A error: {str(e)}")


# Job Management Functions
async def update_job_progress(job_id: str, progress: float, step: str):
    """Actualizar el porcentaje de progreso de un job en SQLite y notificar al cliente via WebSocket si esta conectado."""
    job_data = persistence_manager.get_job(job_id)
    if job_data:
        job_data.update({
            'progress': progress,
            'current_step': step,
            'status': 'processing' if progress < 100 else job_data['status']
        })
        persistence_manager.save_job(job_data)

        # Notificar WebSocket si existe
        if job_id in WEBSOCKET_CONNECTIONS:
            try:
                await WEBSOCKET_CONNECTIONS[job_id].send_json({
                    "job_id": job_id,
                    "progress": progress,
                    "current_step": step,
                    "timestamp": datetime.now().isoformat()
                })
            except:
                # Conexión cerrada, remover
                WEBSOCKET_CONNECTIONS.pop(job_id, None)

async def process_classification_job(job_id: str, factura_dict: Dict):
    """Procesar un job de clasificacion fiscal en background.

    Ejecuta secuencialmente: busqueda AEAT -> busqueda SmartDoc (opcional) ->
    fusion de resultados -> clasificacion fiscal. Actualiza el progreso via
    WebSocket y persiste el resultado o error en SQLite.
    """
    job_wall_start = time.perf_counter()
    logger.info(f"[JOB {job_id}] ⏱️ Inicio de procesamiento asíncrono")
    with JOB_LOCK:
        ACTIVE_JOBS.add(job_id)

    try:
        # Marcar como iniciado
        job_data = persistence_manager.get_job(job_id)
        job_data.update({
            'status': 'processing',
            'started_at': datetime.now().isoformat()
        })
        persistence_manager.save_job(job_data)

        await update_job_progress(job_id, 10.0, "Validating invoice structure")

        # Paso 1: AEAT
        logger.info(f"[JOB {job_id}] 🔍 Inicio búsqueda AEAT")
        aeat_stage_start = time.perf_counter()
        await update_job_progress(job_id, 30.0, "Searching AEAT database")
        try:
            aeat_results = await get_aeat_results_with_cache(factura_dict, allow_wait=True)
            logger.info(
                f"[JOB {job_id}] ✅ Búsqueda AEAT completada en {time.perf_counter() - aeat_stage_start:.2f}s"
            )
        except HTTPException as he:
            if he.status_code == 429:
                logger.warning(
                    f"[JOB {job_id}] ⚠️ Búsqueda AEAT rechazada por límite concurrente tras "
                    f"{time.perf_counter() - aeat_stage_start:.2f}s"
                )
                raise he
            raise _safe_http_error(
                component="aeat_scraper",
                stage="aeat_search",
                error_code="AEAT_SEARCH_ERROR",
                user_message="Servicio AEAT no disponible temporalmente.",
                status_code=500,
                retryable=True,
                exc=he,
            )
        except Exception as e:
            logger.error(
                f"[JOB {job_id}] ❌ Error inesperado durante AEAT tras "
                f"{time.perf_counter() - aeat_stage_start:.2f}s: {e}"
            )
            raise _safe_http_error(
                component="aeat_scraper",
                stage="aeat_search",
                error_code="AEAT_SEARCH_ERROR",
                user_message="Servicio AEAT no disponible temporalmente.",
                status_code=500,
                retryable=True,
                exc=e,
            )

        # Paso 2: SmartDoc
        await update_job_progress(job_id, 60.0, "Searching SmartDoc knowledge base")
        smartdoc_results = None
        if SMARTDOC_ENRICHMENT_ENABLED and AVAILABLE_MODULES["smartdoc"] and global_chunk_metadata:
            logger.info(f"[JOB {job_id}] 📚 Inicio búsqueda SmartDoc")
            smartdoc_stage_start = time.perf_counter()
            try:
                search_query = extract_search_query_from_factura(factura_dict)
                if search_query.strip():
                    smartdoc_results = await search_smartdoc_for_fiscal_async(search_query)
                    logger.info(
                        f"[JOB {job_id}] ✅ SmartDoc completado en "
                        f"{time.perf_counter() - smartdoc_stage_start:.2f}s"
                    )
                else:
                    logger.info(f"[JOB {job_id}] ℹ️ SmartDoc omitido: consulta vacía")
            except Exception as e:
                logger.error(
                    f"[JOB {job_id}] ❌ Error SmartDoc tras "
                    f"{time.perf_counter() - smartdoc_stage_start:.2f}s: {e}"
                )
                raise _safe_http_error(
                    component="smartdoc",
                    stage="smartdoc_search",
                    error_code="SMARTDOC_SEARCH_ERROR",
                    user_message="Repositorio de conocimiento no disponible temporalmente.",
                    status_code=500,
                    retryable=True,
                    exc=e,
                )

        # Paso 3: Merge y clasificar
        logger.info(f"[JOB {job_id}] 🔗 Fusión de resultados")
        merge_stage_start = time.perf_counter()
        await update_job_progress(job_id, 80.0, "Merging results and classifying")
        try:
            # Offload CPU-bound merge to a thread to avoid blocking the event loop
            merged_results = await asyncio.to_thread(
                merge_aeat_and_smartdoc_results, aeat_results, smartdoc_results
            )
            logger.info(
                f"[JOB {job_id}] ✅ Fusión completada en {time.perf_counter() - merge_stage_start:.2f}s"
            )
        except Exception as e:
            logger.error(
                f"[JOB {job_id}] ❌ Error fusionando resultados tras "
                f"{time.perf_counter() - merge_stage_start:.2f}s: {e}"
            )
            raise _safe_http_error(
                component="fiscal_classifier",
                stage="merge_results",
                error_code="RESULTS_MERGE_ERROR",
                user_message="No se pudo combinar la información disponible para clasificar.",
                status_code=500,
                retryable=False,
                exc=e,
            )

        try:
            # Construcción y clasificación potencialmente CPU-bound → ejecutar en hilo
            def _classify_sync():
                classify_start = time.perf_counter()
                classifier = EnhancedFiscalClassifier(merged_results)
                result = classifier.classify_expense_with_precedents(factura_dict)
                logger.info(
                    f"[JOB {job_id}] 🧠 Clasificación completada en {time.perf_counter() - classify_start:.2f}s"
                )
                return result

            resultado = await asyncio.to_thread(_classify_sync)
        except ValueError as e:
            raise _safe_http_error(
                component="fiscal_classifier",
                stage="validation",
                error_code="VALIDATION_ERROR",
                user_message="La estructura de la factura no es válida.",
                status_code=400,
                retryable=False,
                exc=e,
            )
        except Exception as e:
            raise _safe_http_error(
                component="fiscal_classifier",
                stage="classification",
                error_code="CLASSIFICATION_ERROR",
                user_message="Error interno durante la clasificación.",
                status_code=500,
                retryable=True,
                exc=e,
            )

        await update_job_progress(job_id, 100.0, "Classification completed")

        # Completar job
        end_time = datetime.now()
        start_time = datetime.fromisoformat(job_data['started_at'])
        duration = int((end_time - start_time).total_seconds())

        job_data.update({
            'status': 'completed',
            'completed_at': end_time.isoformat(),
            'result_data': resultado,
            'actual_duration': duration,
            'progress': 100.0,
            'current_step': 'Completed'
        })
        persistence_manager.save_job(job_data)

        logger.info(f"✅ Job {job_id} completed in {duration}s")
        logger.info(f"[JOB {job_id}] ⏱️ Tiempo total {time.perf_counter() - job_wall_start:.2f}s")

    except HTTPException as he:
        # Store sanitized error info from our helper
        job_data = persistence_manager.get_job(job_id) or {}
        detail = he.detail if isinstance(he.detail, dict) else {"message": "Error"}
        logger.error(f"❌ Job {job_id} failed [{detail.get('correlation_id','-')}] {detail.get('component')}::{detail.get('stage')} ({detail.get('error_code')})")

        job_data.update({
            'status': 'failed',
            'completed_at': datetime.now().isoformat(),
            # Store sanitized message only
            'error_message': detail.get('message', 'Error'),
            'error_code': detail.get('error_code', 'ERROR'),
            'component': detail.get('component', 'unknown'),
            'stage': detail.get('stage', 'unknown'),
            'correlation_id': detail.get('correlation_id'),
            'progress': 0.0,
            'current_step': f"Error: {detail.get('message', 'Error')}"
        })
        persistence_manager.save_job(job_data)
    except Exception as e:
        # Fallback unexpected error
        he = _safe_http_error(
            component="fiscal_classifier",
            stage="unknown",
            error_code="UNEXPECTED_ERROR",
            user_message="Se produjo un error no esperado durante el procesamiento.",
            status_code=500,
            retryable=True,
            exc=e,
        )
        job_data = persistence_manager.get_job(job_id) or {}
        detail = he.detail if isinstance(he.detail, dict) else {"message": "Error"}
        job_data.update({
            'status': 'failed',
            'completed_at': datetime.now().isoformat(),
            'error_message': detail.get('message', 'Error'),
            'error_code': detail.get('error_code', 'ERROR'),
            'component': detail.get('component', 'unknown'),
            'stage': detail.get('stage', 'unknown'),
            'correlation_id': detail.get('correlation_id'),
            'progress': 0.0,
            'current_step': f"Error: {detail.get('message', 'Error')}"
        })
        persistence_manager.save_job(job_data)

    finally:
        with JOB_LOCK:
            ACTIVE_JOBS.discard(job_id)

        # Cerrar WebSocket si existe
        if job_id in WEBSOCKET_CONNECTIONS:
            try:
                await WEBSOCKET_CONNECTIONS[job_id].close()
            except:
                pass
            finally:
                WEBSOCKET_CONNECTIONS.pop(job_id, None)
# ------------------------------------------------------------------------------
# Startup Event
# ------------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Inicializar todos los componentes al arrancar el servidor.

    Reconstruye variables desde SQLite, inicializa el scraper AEAT singleton
    y el clasificador fiscal, y limpia jobs expirados.
    """
    global fiscal_classifier

    logger.info("🚀 Starting Unified API Server v4.0 with SQLite Persistence...")

    # Reconstruir variables desde SQLite
    reconstruct_variables_from_sqlite()

    # Inicializar AEAT Scraper Singleton
    if AVAILABLE_MODULES["fiscal_classifier"]:
        logger.info("🔄 Initializing AEAT Scraper Singleton...")
        scraper_success = initialize_aeat_scraper(SCRAPER_CONFIG)
        if scraper_success:
            scraper = get_aeat_scraper()
            stats = scraper.get_stats()
            logger.info(f"✅ AEAT Scraper Singleton initialized (Pool size: {stats['pool_size']})")
        else:
            logger.error("❌ Failed to initialize AEAT Scraper Singleton")
            AVAILABLE_MODULES["fiscal_classifier"] = False

    # Initialize fiscal classifier if available
    if AVAILABLE_MODULES["fiscal_classifier"]:
        try:
            fiscal_classifier = EnhancedFiscalClassifier()
            logger.info("✅ Fiscal Classifier initialized successfully (basic mode)")
        except Exception as e:
            logger.error(f"❌ Error initializing Fiscal Classifier: {e}")
            AVAILABLE_MODULES["fiscal_classifier"] = False
            fiscal_classifier = None

    logger.info(f"📊 Available modules: {[k for k, v in AVAILABLE_MODULES.items() if v]}")
    logger.info("🌐 Server ready!")

    persistence_manager.cleanup_expired_jobs()
    logger.info("🧹 Expired jobs cleaned up on startup")


@app.on_event("shutdown")
async def shutdown_event():
    """Persistir el estado en SQLite y cerrar el pool del scraper AEAT al detener el servidor."""
    logger.info("💾 Saving state to SQLite before shutdown...")
    save_all_to_sqlite()

    # Cerrar AEAT Scraper Singleton
    logger.info("🔄 Shutting down AEAT Scraper Singleton...")
    shutdown_aeat_scraper()

    logger.info("👋 Server shutdown complete")


# ------------------------------------------------------------------------------
# System Endpoints
# ------------------------------------------------------------------------------

@app.get("/", response_model=SystemInfo)
async def root():
    """Endpoint raiz que devuelve informacion general del servicio, modulos activos y estado de persistencia."""
    db_size = 0
    try:
        db_size = os.path.getsize(DATABASE_PATH) / (1024 * 1024)  # MB
    except:
        pass

    return SystemInfo(
        service="Unified API Server",
        version="4.0.0",
        description="SmartDoc Document Management + Fiscal Classification AEAT con Persistencia SQLite",
        available_modules=AVAILABLE_MODULES,
        documentation={
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        endpoints={
            "health": "/health",
            "capabilities": "/capabilities",
            "persistence": "/smartdoc/persistence_stats"
        },
        timestamp=datetime.now().isoformat(),
        persistence={
            "database_path": DATABASE_PATH,
            "database_size_mb": round(db_size, 2),
            "documents_in_memory": len(documents),
            "persistence_enabled": True
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check integral de todos los subsistemas (vLLM, SmartDoc, scraper AEAT, clasificador fiscal)."""
    components = {}
    overall_status = "healthy"

    # SmartDoc health check (no bloquear event loop)
    if AVAILABLE_MODULES["smartdoc"]:
        try:
            # Ejecutar la llamada al backend vLLM en un hilo con timeout corto
            async def _ping_vllm():
                return await asyncio.to_thread(
                    vllm_client.generate, "Hello", "Respond with 'OK'"
                )
            try:
                test_response = await asyncio.wait_for(_ping_vllm(), timeout=2.0)
                vllm_status = "connected" if test_response else "error"
            except Exception as e:
                vllm_status = f"error: {str(e)}"
                overall_status = "degraded"
        except Exception as e:
            vllm_status = f"error: {str(e)}"
            overall_status = "degraded"

        global_chunks = len(global_chunk_metadata) if global_chunk_metadata else 0
        unique_docs_in_global = len(
            set(meta['doc_id'] for meta in global_chunk_metadata)) if global_chunk_metadata else 0

        db_size = 0
        try:
            db_size = os.path.getsize(DATABASE_PATH) / (1024 * 1024)  # MB
        except:
            pass

        components["smartdoc"] = {
            "status": "healthy" if vllm_status == "connected" else "degraded",
            "vllm_server": vllm_client.base_url,
            "vllm_model": vllm_client.model_name,
            "vllm_status": vllm_status,
            "total_documents": len(documents),
            "global_index_chunks": global_chunks,
            "unique_documents_in_global": unique_docs_in_global,
            "duplicate_detection": True,
            "persistence": {
                "database_path": DATABASE_PATH,
                "database_size_mb": round(db_size, 2),
                "persistence_enabled": True
            }
        }

    # Fiscal classifier health check CON scraper singleton
    if AVAILABLE_MODULES["fiscal_classifier"]:
        try:
            # Health check del fiscal classifier
            fiscal_status = "healthy" if fiscal_classifier is not None else "error"

            # Health check del scraper singleton
            scraper = get_aeat_scraper()
            scraper_health = scraper.health_check()
            scraper_stats = scraper.get_stats()

            components["fiscal_classifier"] = {
                "status": fiscal_status,
                "classifier_initialized": fiscal_classifier is not None,
                "scraper_singleton": {
                    "status": scraper_health["status"],
                    "pool_active": scraper_health["pool_active"],
                    "pool_size": scraper_health["pool_size"],
                    "total_requests": scraper_stats["total_requests"],
                    "successful_requests": scraper_stats["successful_requests"],
                    "error_rate": f"{scraper_stats.get('error_rate', 0):.2f}%",
                    "avg_response_time": f"{scraper_stats['avg_response_time']:.2f}s",
                    "uptime_hours": f"{scraper_stats.get('uptime_hours', 0):.2f}h",
                    "pool_restarts": scraper_stats["pool_restarts"]
                },
                "capabilities": [
                    "AEAT codes G01-G46",
                    "Deductibility analysis",
                    "Singleton scraper pool",
                    f"Pool size: {scraper_stats['pool_size']} drivers"
                ]
            }

            # Marcar como degraded si el scraper no está healthy
            if scraper_health["status"] != "healthy":
                overall_status = "degraded"

        except Exception as e:
            components["fiscal_classifier"] = {
                "status": "error",
                "error": str(e)
            }
            overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version="4.0.0",
        timestamp=datetime.now().isoformat(),
        components=components
    )

@app.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities():
    """Listar las capacidades del sistema y los endpoints disponibles segun los modulos activos."""
    endpoints = {
        "system": {
            "GET /": "Root information with persistence stats",
            "GET /health": "System health check with persistence and scraper singleton",
            "GET /capabilities": "System capabilities",
            "GET /docs": "API documentation",
            "GET /redoc": "Alternative API documentation"
        }
    }

    if AVAILABLE_MODULES["smartdoc"]:
        endpoints["smartdoc"] = {
            "POST /smartdoc/upload": "Upload and process documents with SQLite persistence",
            "GET /smartdoc/documents": "List all documents from SQLite",
            "GET /smartdoc/document/{id}/summary": "Generate document summary",
            "POST /smartdoc/document/{id}/query": "Query specific document",
            "POST /smartdoc/query_all": "Query across all documents with persistence",
            "GET /smartdoc/persistence_stats": "Get SQLite persistence statistics",
            "POST /smartdoc/backup": "Create manual backup",
            "DELETE /smartdoc/document/{id}": "Delete document with persistence cleanup"
        }

    if AVAILABLE_MODULES["fiscal_classifier"]:
        endpoints["fiscal_classifier"] = {
            "POST /fiscal/classify": "Classify invoice using shared scraper pool",
            "POST /fiscal/validate": "Validate invoice structure",
            "GET /fiscal/scraper_stats": "Get scraper singleton statistics",
            "GET /fiscal/scraper_health": "Check scraper singleton health",
            "POST /fiscal/restart_scraper": "Restart scraper pool manually",
            "GET /fiscal/scraper_performance": "Get performance metrics and recommendations"
        }

    return CapabilitiesResponse(
        version="4.0.0",
        available_modules=[k for k, v in AVAILABLE_MODULES.items() if v],
        endpoints=endpoints,
        timestamp=datetime.now().isoformat()
    )


# ------------------------------------------------------------------------------
# SmartDoc Endpoints (unchanged from original)
# ------------------------------------------------------------------------------

@app.post("/smartdoc/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Subir y procesar un documento (PDF o texto) con deteccion de duplicados.

    Extrae texto, genera embeddings, crea indice FAISS individual,
    anade al indice global y persiste todo en SQLite.
    """
    if not AVAILABLE_MODULES["smartdoc"]:
        raise HTTPException(status_code=503, detail="SmartDoc module not available")

    global doc_id_counter

    text = extract_text(file)
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text found.")

    fingerprint = create_document_fingerprint(text)
    doc_id = doc_id_counter
    duplicate_info = handle_duplicate_document(fingerprint, doc_id, file.filename)

    chunks = chunk_text(text, chunk_size=1000)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document processing resulted in no content.")

    embeddings = create_embeddings(chunks)
    index = create_faiss_index(embeddings)

    # Crear documento en memoria
    doc_data = {
        "filename": file.filename,
        "text": text,
        "chunks": chunks,
        "embeddings": embeddings,
        "fingerprint": fingerprint,
        "upload_timestamp": datetime.now().isoformat()
    }

    documents[doc_id] = doc_data
    faiss_indexes[doc_id] = index

    # Guardar en SQLite
    try:
        persistence_manager.save_document(doc_id, doc_data)
        persistence_manager.save_faiss_index(doc_id, index)
        persistence_status = "saved"
    except Exception as e:
        logger.error(f"Error saving to SQLite: {e}")
        persistence_status = f"error: {str(e)}"

    add_to_global_index(embeddings, doc_id, file.filename, chunks)

    if not duplicate_info["duplicate_detected"]:
        doc_id_counter += 1
        persistence_manager.save_doc_id_counter(doc_id_counter)

    message = (
        f"Document processed successfully. Replaced previous version (ID: {duplicate_info['old_doc_id']}) with new version. Model: {vllm_client.model_name}"
        if duplicate_info["duplicate_detected"]
        else f"Document uploaded and processed successfully using vLLM backend. Model: {vllm_client.model_name}")

    return UploadResponse(
        doc_id=doc_id,
        message=message,
        duplicate_info=duplicate_info,
        persistence_status=persistence_status
    )


@app.get("/smartdoc/documents", response_model=List[DocumentInfo])
async def list_documents():
    """Listar todos los documentos subidos con sus metadatos basicos."""
    if not AVAILABLE_MODULES["smartdoc"]:
        raise HTTPException(status_code=503, detail="SmartDoc module not available")

    docs_info = []
    for doc_id, data in documents.items():
        docs_info.append(DocumentInfo(
            doc_id=doc_id,
            filename=data.get("filename", "Unknown"),
            text_length=len(data.get("text", "")),
            num_chunks=len(data.get("chunks", [])),
            upload_timestamp=data.get("upload_timestamp", "Unknown")
        ))
    return docs_info


@app.get("/smartdoc/persistence_stats", response_model=PersistenceStats)
async def get_persistence_stats():
    """Obtener estadisticas de la capa de persistencia SQLite (tamano de BD, total de documentos y chunks)."""
    if not AVAILABLE_MODULES["smartdoc"]:
        raise HTTPException(status_code=503, detail="SmartDoc module not available")

    db_size = 0
    try:
        db_size = os.path.getsize(DATABASE_PATH) / (1024 * 1024)  # MB
    except:
        pass

    return PersistenceStats(
        database_path=DATABASE_PATH,
        total_documents=len(documents),
        total_chunks=len(global_chunk_metadata),
        database_size_mb=round(db_size, 2),
        last_backup=datetime.now().isoformat()
    )


@app.post("/smartdoc/rebuild_global_index")
async def rebuild_global_faiss_index():
    """Reconstruir el indice FAISS global y sus metadatos a partir de los embeddings almacenados.

    Itera sobre todos los documentos en memoria (reconstruidos desde SQLite),
    concatena los embeddings, construye un nuevo IndexFlatL2, recalcula
    global_chunk_metadata con ordenamiento consistente y persiste ambos en SQLite.
    """
    global global_faiss_index, global_chunk_metadata

    if not AVAILABLE_MODULES["smartdoc"]:
        raise HTTPException(status_code=503, detail="SmartDoc module not available")

    total_docs = len(documents)
    used_docs = 0
    skipped_docs = []
    total_vectors = 0
    dimension = None

    all_embeddings = []
    new_metadata = []

    try:
        for doc_id, doc in documents.items():
            emb = doc.get('embeddings')
            chunks = doc.get('chunks', [])

            # Require 2D non-empty embeddings
            if emb is None or not hasattr(emb, 'shape') or len(emb.shape) != 2 or emb.shape[0] == 0:
                skipped_docs.append({"doc_id": doc_id, "reason": "missing_or_invalid_embeddings"})
                continue

            if dimension is None:
                dimension = emb.shape[1]
            elif emb.shape[1] != dimension:
                skipped_docs.append({"doc_id": doc_id, "reason": f"dim_mismatch:{emb.shape[1]}!= {dimension}"})
                continue

            n_vecs = emb.shape[0]
            if len(chunks) != n_vecs:
                min_len = min(len(chunks), n_vecs)
                if min_len == 0:
                    skipped_docs.append({"doc_id": doc_id, "reason": "no_chunks_or_vectors"})
                    continue
                emb = emb[:min_len]
                chunks = chunks[:min_len]

            all_embeddings.append(emb)

            for i, chunk_text in enumerate(chunks):
                new_metadata.append({
                    'doc_id': doc_id,
                    'filename': doc.get('filename', f'doc_{doc_id}'),
                    'chunk_index': i,
                    'chunk_text': chunk_text,
                    'global_index': len(new_metadata)
                })

            used_docs += 1
            total_vectors += emb.shape[0]

        if not all_embeddings or dimension is None:
            raise HTTPException(status_code=400, detail="No valid embeddings found to build global index")

        embeddings_array = np.vstack(all_embeddings)
        new_index = faiss.IndexFlatL2(dimension)
        new_index.add(embeddings_array)

        global_faiss_index = new_index
        global_chunk_metadata = new_metadata

        persistence_manager.save_global_faiss_index(global_faiss_index)
        persistence_manager.save_global_chunk_metadata(global_chunk_metadata)

        return {
            "message": "Global FAISS index rebuilt successfully",
            "documents_total": total_docs,
            "documents_used": used_docs,
            "documents_skipped": skipped_docs,
            "vectors_total": int(total_vectors),
            "dimension": int(dimension),
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rebuilding global index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild global index: {str(e)}")


@app.post("/smartdoc/backup")
async def create_manual_backup():
    """Crear un backup manual forzando la escritura de todos los datos en memoria a SQLite."""
    if not AVAILABLE_MODULES["smartdoc"]:
        raise HTTPException(status_code=503, detail="SmartDoc module not available")

    try:
        save_all_to_sqlite()
        return {
            "message": "Manual backup completed successfully",
            "timestamp": datetime.now().isoformat(),
            "documents_saved": len(documents),
            "chunks_saved": len(global_chunk_metadata)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {str(e)}")


@app.delete("/smartdoc/document/{doc_id}")
async def delete_document(doc_id: int):
    """Eliminar un documento de memoria y SQLite, incluyendo su indice FAISS, fingerprint y chunks globales."""
    if not AVAILABLE_MODULES["smartdoc"]:
        raise HTTPException(status_code=503, detail="SmartDoc module not available")

    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    filename = documents[doc_id]['filename']
    fingerprint = documents[doc_id]['fingerprint']

    # Eliminar de memoria
    remove_document_from_global_index(doc_id)
    del documents[doc_id]

    if doc_id in faiss_indexes:
        del faiss_indexes[doc_id]

    # Eliminar fingerprint
    if fingerprint in document_fingerprints:
        del document_fingerprints[fingerprint]
        persistence_manager.save_document_fingerprints(document_fingerprints)

    # Eliminar de SQLite
    try:
        persistence_manager.delete_document(doc_id)
        persistence_status = "deleted from SQLite"
    except Exception as e:
        logger.error(f"Error deleting from SQLite: {e}")
        persistence_status = f"SQLite deletion error: {str(e)}"

    return {
        "message": f"Document '{filename}' (ID: {doc_id}) deleted successfully",
        "doc_id": doc_id,
        "filename": filename,
        "persistence_status": persistence_status,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/smartdoc/document/{doc_id}/summary", response_model=SummaryResponse)
async def summarize_document(doc_id: int):
    """Generar un resumen inteligente de un documento usando vLLM."""
    if not AVAILABLE_MODULES["smartdoc"]:
        raise HTTPException(status_code=503, detail="SmartDoc module not available")

    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    text = documents[doc_id]["text"]

    try:
        summary = summarize_with_vllm(text)
        return SummaryResponse(doc_id=doc_id, summary=summary)
    except Exception as e:
        logger.error(f"Summarization error for doc {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating summary. Please try again.")


@app.post("/smartdoc/document/{doc_id}/query", response_model=QueryResponse)
async def query_document(doc_id: int, query_request: QueryRequest):
    """Responder preguntas sobre un documento especifico usando busqueda semantica FAISS + vLLM."""
    if not AVAILABLE_MODULES["smartdoc"]:
        raise HTTPException(status_code=503, detail="SmartDoc module not available")

    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    query = query_request.query

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    index = faiss_indexes[doc_id]
    k = 6
    distances, indices = index.search(query_embedding, k)

    chunks = documents[doc_id]["chunks"]
    retrieved_chunks = []

    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    if not retrieved_chunks:
        raise HTTPException(status_code=500, detail="Unable to retrieve relevant context for the query.")

    try:
        answer = answer_with_vllm(query, retrieved_chunks)

        return QueryResponse(
            doc_id=doc_id,
            query=query,
            answer=answer,
            context_chunks=retrieved_chunks
        )
    except Exception as e:
        logger.error(f"Q&A error for doc {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating answer. Please try again.")


@app.post("/smartdoc/query_all", response_model=GlobalQueryResponse)
async def query_all_documents(query_request: QueryRequest):
    """Buscar y responder preguntas a traves de TODOS los documentos usando el indice FAISS global.

    Si el indice global no existe o no devuelve resultados, intenta reconstruirlo
    automaticamente antes de reintentar la busqueda.
    """
    if not AVAILABLE_MODULES["smartdoc"]:
        raise HTTPException(status_code=503, detail="SmartDoc module not available")

    if not global_chunk_metadata:
        raise HTTPException(status_code=404, detail="No documents found in the database.")

    query = query_request.query

    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # Proactive rebuild if global index is missing
    global global_faiss_index
    if global_faiss_index is None:
        try:
            await rebuild_global_faiss_index()
        except Exception as e:
            logger.warning(f"Global index rebuild failed before search: {e}")

    k = 10
    retrieved_chunks, metadata = search_global_index(query_embedding, k)

    # If no results, attempt a one-time rebuild and retry
    if not retrieved_chunks:
        try:
            rebuild_info = await rebuild_global_faiss_index()
            logger.info(f"🔄 Rebuilt global FAISS index on-demand: {rebuild_info}")
            # Retry search once
            retrieved_chunks, metadata = search_global_index(query_embedding, k)
        except Exception as e:
            logger.error(f"On-demand global index rebuild failed: {e}")

    if not retrieved_chunks:
        raise HTTPException(status_code=500, detail="Unable to retrieve relevant context from any document.")

    try:
        answer = answer_global_with_vllm(query, retrieved_chunks, metadata)

        sources_info = []
        seen_docs = {}

        for meta in metadata:
            doc_key = f"{meta['doc_id']}_{meta['filename']}"
            if doc_key not in seen_docs:
                seen_docs[doc_key] = {
                    "doc_id": meta['doc_id'],
                    "filename": meta['filename'],
                    "chunks_used": 0
                }
            seen_docs[doc_key]["chunks_used"] += 1

        sources_info = list(seen_docs.values())

        return GlobalQueryResponse(
            query=query,
            answer=answer,
            context_chunks=retrieved_chunks,
            sources=sources_info,
            total_documents_searched=len(set(meta['doc_id'] for meta in metadata))
        )

    except Exception as e:
        logger.error(f"Global Q&A error: {e}")
        raise HTTPException(status_code=500, detail="Error generating global answer. Please try again.")


@app.get("/fiscal/scraper_health")
async def scraper_health_check():
    """Health check dedicado para el scraper AEAT singleton (estado del pool de navegadores)."""
    if not AVAILABLE_MODULES["fiscal_classifier"]:
        raise HTTPException(status_code=503, detail="Fiscal classifier module not available")

    try:
        scraper = get_aeat_scraper()
        health = scraper.health_check()
        return health
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking scraper health: {str(e)}")


@app.post("/fiscal/restart_scraper")
async def restart_scraper_pool():
    """Reiniciar manualmente el pool de navegadores del scraper AEAT."""
    if not AVAILABLE_MODULES["fiscal_classifier"]:
        raise HTTPException(status_code=503, detail="Fiscal classifier module not available")

    try:
        scraper = get_aeat_scraper()
        success = scraper.restart_pool()

        return {
            "success": success,
            "message": "Scraper pool restarted successfully" if success else "Failed to restart scraper pool",
            "timestamp": datetime.now().isoformat(),
            "new_stats": scraper.get_stats() if success else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restarting scraper: {str(e)}")


@app.get("/fiscal/scraper_performance")
async def get_scraper_performance():
    """Obtener metricas de rendimiento del scraper AEAT, concurrencia y recomendaciones de optimizacion."""
    if not AVAILABLE_MODULES["fiscal_classifier"]:
        raise HTTPException(status_code=503, detail="Fiscal classifier module not available")

    try:
        scraper = get_aeat_scraper()
        stats = scraper.get_stats()

        # Calcular métricas de rendimiento
        error_rate = stats.get('error_rate', 0)
        avg_response_time = stats['avg_response_time']
        requests_per_hour = stats.get('requests_per_hour', 0)

        # Generar recomendaciones
        recommendations = []
        if error_rate > 10:
            recommendations.append("High error rate detected - consider restarting pool")
        if avg_response_time > 10:
            recommendations.append("Slow response times - consider increasing pool size")
        if requests_per_hour > 100:
            recommendations.append("High request volume - monitor pool performance")

        performance_grade = "A"
        if error_rate > 5 or avg_response_time > 5:
            performance_grade = "B"
        if error_rate > 10 or avg_response_time > 10:
            performance_grade = "C"

        # AEAT concurrency metrics
        with AEAT_METRICS_LOCK:
            if AEAT_WAIT_COUNT > 0:
                avg_wait_ms = (AEAT_WAIT_TOTAL_SECONDS / AEAT_WAIT_COUNT) * 1000.0
            else:
                avg_wait_ms = 0.0
            aeat_metrics = {
                "max_concurrent": AEAT_MAX_CONCURRENT,
                "active_now": AEAT_ACTIVE_REQUESTS,
                "wait_count": AEAT_WAIT_COUNT,
                "avg_wait_ms": round(avg_wait_ms, 1),
                "max_wait_ms": int(AEAT_WAIT_MAX_SECONDS * 1000),
                "rejections_429": AEAT_429_REJECTIONS,
                "cache_entries": len(AEAT_CACHE),
            }

        return {
            "performance_metrics": {
                "error_rate_percent": f"{error_rate:.2f}%",
                "avg_response_time_seconds": f"{avg_response_time:.2f}s",
                "requests_per_hour": f"{requests_per_hour:.1f}",
                "performance_grade": performance_grade,
                "uptime_hours": f"{stats.get('uptime_hours', 0):.2f}h"
            },
            "aeat_concurrency": aeat_metrics,
            "recommendations": recommendations,
            "pool_config": {
                "pool_size": stats['pool_size'],
                "pool_restarts": stats['pool_restarts']
            },
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting performance metrics: {str(e)}")


# ------------------------------------------------------------------------------
# Fiscal Classifier Endpoints
# ------------------------------------------------------------------------------

"""
@app.post("/fiscal/classify", response_model=FiscalClassificationResponse)
async def classify_fiscal_expense(factura: FacturaRequest, resp: Response):
    if not AVAILABLE_MODULES["fiscal_classifier"]:
        raise HTTPException(status_code=503, detail="Fiscal classifier module not available")

    if fiscal_classifier is None:
        raise HTTPException(status_code=503, detail="Fiscal classifier not initialized")

    try:
        factura_dict = factura.model_dump(exclude_none=True)

        # 🎯 USAR SINGLETON EN LUGAR DE CREAR NUEVO SCRAPER
        logger.info("🔍 Starting AEAT classification with singleton pool")

        # Usar el scraper singleton de forma async y thread-safe
        results = await async_search_comprehensive(factura_dict)

        # Continuar con la clasificación
        classifier = EnhancedFiscalClassifier(results)
        resultado = classifier.classify_expense_with_precedents(factura_dict)

        logger.info(
            f"✅ Classification completed - Code: {resultado.get('clasificacion', {}).get('codigo_principal', 'N/A')}")

        return FiscalClassificationResponse(**resultado)

    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Internal error in classification: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
"""


@app.post("/fiscal/classify", response_model=FiscalClassificationResponse)
async def classify_fiscal_expense(factura: FacturaRequest):
    """Clasificar una factura de forma sincronica usando internamente el flujo asincrono.

    Crea un job de clasificacion en background y espera hasta 240 segundos a
    que finalice. Si el job completa a tiempo, devuelve el resultado como
    ``FiscalClassificationResponse``. Si supera el timeout, devuelve el payload
    del job asincrono (job_id, URLs de estado y resultado) para que el cliente
    pueda consultarlo posteriormente.
    """
    if not AVAILABLE_MODULES["fiscal_classifier"]:
        raise HTTPException(status_code=503, detail="Fiscal classifier module not available")

    if fiscal_classifier is None:
        raise HTTPException(status_code=503, detail="Fiscal classifier not initialized")

    try:
        # 1) Lanzar el job asíncrono reutilizando la lógica del endpoint
        # Nota: si llamamos al endpoint async directo con un BackgroundTasks creado ad-hoc,
        # FastAPI no ejecutará esas tareas. Por ello, creamos el job y encolamos manualmente.
        if not AVAILABLE_MODULES["fiscal_classifier"]:
            raise HTTPException(status_code=503, detail="Fiscal classifier module not available")

        # Verificar límite de jobs concurrentes (misma política que /fiscal/classify_async)
        with JOB_LOCK:
            if len(ACTIVE_JOBS) >= MAX_CONCURRENT_JOBS:
                raise HTTPException(status_code=429, detail="Too many concurrent jobs. Please try again later.")

        factura_dict = factura.model_dump(exclude_none=True)

        # Validación básica igual que en async
        if fiscal_classifier:
            fiscal_classifier.validate_enhanced_factura_input(factura_dict)

        job_id = str(uuid.uuid4())
        job_data = {
            'job_id': job_id,
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'progress': 0.0,
            'current_step': 'Queued for processing',
            'factura_data': factura_dict,
            'estimated_duration': 45
        }
        persistence_manager.save_job(job_data)

        # Encolar explícitamente la tarea en el event loop actual
        asyncio.create_task(process_classification_job(job_id, factura_dict))

        async_payload = {
            "job_id": job_id,
            "status": "pending",
            "estimated_time": "30-60 seconds",
            "websocket_url": f"/fiscal/ws/{job_id}",
            "status_url": f"/fiscal/status/{job_id}",
            "result_url": f"/fiscal/result/{job_id}"
        }

        if not job_id:
            # Algo salió mal creando el job
            raise HTTPException(status_code=500, detail="Error creating classification job")

        # 2) Esperar hasta 40s a que el job finalice
        deadline = time.monotonic() + 240.0
        while time.monotonic() < deadline:
            job_data = persistence_manager.get_job(job_id)
            if job_data:
                status = job_data.get('status')
                if status == 'completed' and job_data.get('result_data'):
                    # Devolver EXACTO el cuerpo que daría /fiscal/classify (modelo FiscalClassificationResponse)
                    return FiscalClassificationResponse(**job_data['result_data'])
                if status == 'failed':
                    # Propagar error de forma razonable
                    msg = job_data.get('error_message') or 'Classification failed'
                    code = job_data.get('error_code')
                    # Si fue validación, 400; si no, 500
                    status_code = 400 if code == 'VALIDATION_ERROR' else 500
                    raise HTTPException(status_code=status_code, detail=msg)
            await asyncio.sleep(0.5)

        # 3) Timeout: devolver EXACTAMENTE como /fiscal/classify_async
        # Usamos JSONResponse para bypass del response_model y respetar el payload original
        return JSONResponse(content=async_payload, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        # Fallback sanitizado
        raise _safe_http_error(
            component="fiscal_classifier",
            stage="unknown",
            error_code="UNEXPECTED_ERROR",
            user_message="Se produjo un error no esperado. Por favor, inténtalo de nuevo más tarde.",
            status_code=500,
            retryable=True,
            exc=e,
        )


@app.post("/fiscal/classify_async")
async def classify_fiscal_async(factura: FacturaRequest, background_tasks: BackgroundTasks):
    """Iniciar una clasificacion fiscal asincrona como job en background.

    Valida la factura, crea un job con estado 'pending' y lo encola para
    procesamiento. Devuelve el job_id y URLs para consultar estado/resultado
    via polling o WebSocket.
    """
    if not AVAILABLE_MODULES["fiscal_classifier"]:
        raise HTTPException(status_code=503, detail="Fiscal classifier module not available")

    # Verificar límite de jobs concurrentes
    with JOB_LOCK:
        if len(ACTIVE_JOBS) >= MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="Too many concurrent jobs. Please try again later.")

    try:
        factura_dict = factura.model_dump(exclude_none=True)

        # Validar estructura
        if fiscal_classifier:
            fiscal_classifier.validate_enhanced_factura_input(factura_dict)

        # Crear job
        job_id = str(uuid.uuid4())
        job_data = {
            'job_id': job_id,
            'status': 'pending',
            'created_at': datetime.now().isoformat(),
            'progress': 0.0,
            'current_step': 'Queued for processing',
            'factura_data': factura_dict,
            'estimated_duration': 45  # seconds
        }

        persistence_manager.save_job(job_data)

        # Encolar tarea
        background_tasks.add_task(process_classification_job, job_id, factura_dict)

        return {
            "job_id": job_id,
            "status": "pending",
            "estimated_time": "30-60 seconds",
            "websocket_url": f"/fiscal/ws/{job_id}",
            "status_url": f"/fiscal/status/{job_id}",
            "result_url": f"/fiscal/result/{job_id}"
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating async job: {e}")
        raise HTTPException(status_code=500, detail="Error creating classification job")


@app.get("/fiscal/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Obtener el estado actual de un job de clasificacion, incluyendo progreso y tiempo restante estimado."""
    job_data = persistence_manager.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    # Calcular tiempo restante estimado
    estimated_remaining = None
    if job_data['status'] == 'processing' and job_data.get('estimated_duration'):
        if job_data.get('started_at'):
            elapsed = (datetime.now() - datetime.fromisoformat(job_data['started_at'])).total_seconds()
            remaining = max(0, job_data['estimated_duration'] - elapsed)
            estimated_remaining = f"{int(remaining)}s"

    return JobStatusResponse(
        job_id=job_data['job_id'],
        status=job_data['status'],
        progress=job_data['progress'],
        current_step=job_data['current_step'],
        created_at=job_data['created_at'],
        estimated_remaining=estimated_remaining
    )


@app.get("/fiscal/result/{job_id}", response_model=JobResultResponse)
async def get_job_result(job_id: str):
    """Obtener el resultado final de un job de clasificacion completado o fallido."""
    job_data = persistence_manager.get_job(job_id)

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_data['status'] not in ['completed', 'failed']:
        raise HTTPException(status_code=202, detail=f"Job still {job_data['status']}")

    result = None
    if job_data['status'] == 'completed' and job_data.get('result_data'):
        result = FiscalClassificationResponse(**job_data['result_data'])

    return JobResultResponse(
        job_id=job_data['job_id'],
        status=job_data['status'],
        result=result,
        error=job_data.get('error_message'),
        completed_at=job_data.get('completed_at') or '',
        actual_duration=int(job_data.get('actual_duration') or 0)
    )


@app.websocket("/fiscal/ws/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    """WebSocket para recibir actualizaciones en tiempo real del progreso de un job de clasificacion."""
    await websocket.accept()

    # Verificar que el job existe
    job_data = persistence_manager.get_job(job_id)
    if not job_data:
        await websocket.close(code=4004, reason="Job not found")
        return

    # Registrar conexión
    WEBSOCKET_CONNECTIONS[job_id] = websocket

    try:
        # Enviar estado inicial
        await websocket.send_json({
            "job_id": job_id,
            "status": job_data['status'],
            "progress": job_data['progress'],
            "current_step": job_data['current_step'],
            "timestamp": datetime.now().isoformat()
        })

        # Mantener conexión hasta que el job termine
        while True:
            job_data = persistence_manager.get_job(job_id)
            if job_data and job_data['status'] in ['completed', 'failed']:
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job_data['status'],
                    "progress": job_data['progress'],
                    "current_step": job_data['current_step'],
                    "final": True,
                    "timestamp": datetime.now().isoformat()
                })
                break

            # Ping cada 5 segundos
            await asyncio.sleep(5)
            await websocket.send_json({"type": "ping", "timestamp": datetime.now().isoformat()})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        WEBSOCKET_CONNECTIONS.pop(job_id, None)


@app.post("/fiscal/validate", response_model=ValidationResponse)
async def validate_factura_structure(factura: FacturaRequest):
    """Validar la estructura de una factura sin ejecutar la clasificacion fiscal completa."""
    if not AVAILABLE_MODULES["fiscal_classifier"]:
        raise HTTPException(status_code=503, detail="Fiscal classifier module not available")

    try:
        factura_dict = factura.model_dump(exclude_none=True)

        if fiscal_classifier:
            fiscal_classifier.validate_enhanced_factura_input(factura_dict)

        return ValidationResponse(
            valid=True,
            message="Valid invoice structure",
            conceptos_count=len(factura_dict.get('conceptos', [])),
            sections_present=list(factura_dict.keys()),
            timestamp=datetime.now().isoformat()
        )

    except ValueError as e:
        return ValidationResponse(
            valid=False,
            error=str(e),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error validando factura: {e}")
        raise HTTPException(status_code=500, detail="Error interno de validación")


# ------------------------------------------------------------------------------
# New Endpoint: Load External Consultas
# ------------------------------------------------------------------------------

@app.post("/fiscal/load_consultas")
async def load_external_consultas(consultas_data: List[Dict]):
    """Cargar datos de consultas vinculantes externas en el clasificador fiscal."""
    if not AVAILABLE_MODULES["fiscal_classifier"]:
        raise HTTPException(status_code=503, detail="Fiscal classifier module not available")

    if fiscal_classifier is None:
        raise HTTPException(status_code=503, detail="Fiscal classifier not initialized")

    try:
        success = fiscal_classifier.load_consultas_from_data(consultas_data)

        if success:
            return {
                "message": f"Successfully loaded {len(consultas_data)} consultas",
                "status": "success",
                "consultas_loaded": len(consultas_data),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load consultas data")

    except Exception as e:
        logger.error(f"Error loading external consultas: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading consultas: {str(e)}")


@app.post("/fiscal/cleanup_jobs")
async def cleanup_expired_jobs_endpoint():
    """Limpiar manualmente los jobs expirados de la base de datos SQLite."""
    if not AVAILABLE_MODULES["fiscal_classifier"]:
        raise HTTPException(status_code=503, detail="Fiscal classifier module not available")

    persistence_manager.cleanup_expired_jobs()

    return {
        "message": "Expired jobs cleaned up successfully",
        "timestamp": datetime.now().isoformat()
    }
# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("🏛️ Starting Unified API Server v4.0 - BlueBill App con Persistencia SQLite + Scraper Singleton")
    print("📄 SmartDoc Document Management + Fiscal Classification AEAT")
    print(f"📡 vLLM Server: {VLLM_BASE_URL}")
    print(f"🤖 vLLM Model: {VLLM_MODEL}")
    print("🔍 Embedding Model: all-MiniLM-L6-v2 (CPU)")
    print("🌐 Global Index: Enabled for cross-document search")
    print("🔄 Duplicate Detection: Enabled (newest version predominates)")
    print("👆 Fingerprint Method: First 128 + Last 128 characters")
    print("💾 Persistence: SQLite enabled - data survives restarts")
    print(f"🗄️ Database Path: {DATABASE_PATH}")
    print("🚀 Auto-reconstruction: All variables restored from SQLite on startup")
    print("🕷️ AEAT Scraper: Singleton pool with 32 drivers shared across all requests")
    print("🔄 Pool Management: Auto-restart every hour + error recovery")
    print(f"📊 Available modules: {[k for k, v in AVAILABLE_MODULES.items() if v]}")
    print("=" * 90)

    uvicorn.run(app, host="0.0.0.0", port=8001, timeout_keep_alive=2400)
    # uvicorn.run(app, host="0.0.0.0", port=8001, timeout_keep_alive=2400, workers=8, limit_concurrency=1000, backlog=2048)
