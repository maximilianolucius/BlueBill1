"""
Backend de SmartDoc AI con integración vLLM.

Módulo principal del servidor backend que proporciona una API REST (FastAPI) para
la carga, procesamiento, resumen y consulta inteligente de documentos. Utiliza
vLLM como motor de inferencia remoto para generación de resúmenes y respuestas
a preguntas, combinado con SentenceTransformer para embeddings semánticos y
FAISS para búsqueda vectorial eficiente.

Funcionalidades principales:
    - Carga y extracción de texto desde archivos PDF y texto plano.
    - Detección de documentos duplicados mediante huella digital (fingerprint).
    - Segmentación del texto en fragmentos (chunks) con solapamiento.
    - Generación de embeddings y construcción de índices FAISS por documento
      y a nivel global para búsqueda cruzada entre documentos.
    - Resumen automático de documentos mediante vLLM.
    - Respuesta a preguntas (Q&A) sobre documentos individuales o sobre
      toda la base de documentos con atribución de fuentes.
"""

import io
import os
from typing import List, Dict, Any, Iterator
from abc import ABC, abstractmethod

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import PyPDF2  # For PDF text extraction
import numpy as np
from openai import OpenAI

from sentence_transformers import SentenceTransformer  # For computing embeddings
import faiss  # For vector (embedding) search

from utils.generator_model import VLLMClient as VLLMSingleton

# Load environment variables
load_dotenv()

# vLLM Configuration
VLLM_BASE_URL = os.getenv('VLLM_BASE_URL', "http://172.24.250.17:8000/v1")
VLLM_MODEL = os.getenv('VLLM_MODEL', "Qwen3-8B-AWQ")
VLLM_API_KEY = os.getenv('VLLM_API_KEY', "EMPTY")

# ------------------------------------------------------------------------------
# Initialize FastAPI and Configure CORS
# ------------------------------------------------------------------------------
app = FastAPI(
    title="SmartDoc Backend API with vLLM",
    description="A self‑hosted AI document summarizer and Q&A backend using vLLM remote inference.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# In‑Memory Storage
# ------------------------------------------------------------------------------
documents: Dict[int, Dict[str, Any]] = {}
faiss_indexes: Dict[int, faiss.IndexFlatL2] = {}
doc_id_counter = 1

# Global FAISS index for cross-document search
global_faiss_index: faiss.IndexFlatL2 = None
global_chunk_metadata: List[Dict[str, Any]] = []  # Metadata for each chunk in global index

# Document fingerprint tracking for duplicate detection
document_fingerprints: Dict[str, int] = {}  # fingerprint -> doc_id

# ------------------------------------------------------------------------------
# Load Models at Startup
# ------------------------------------------------------------------------------
print("Loading models... (this may take a few moments)")
try:
    # vLLM Client for Summarization and Q&A
    vllm_client = VLLMSingleton()
    print(f"✅ vLLM client initialized: {vllm_client.base_url} - Model: {vllm_client.model_name}")

    # Embedding model: Keep local for fast semantic search - FORCE CPU usage
    import torch

    device = 'cpu'  # Force CPU usage to avoid CUDA issues
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print(f"✅ Embedding model loaded: all-MiniLM-L6-v2 (device: {device})")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def extract_text(file: UploadFile) -> str:
    """Extrae texto de un archivo subido (PDF o texto plano).

    Lee el contenido del archivo y devuelve el texto extraído. Para archivos
    PDF utiliza PyPDF2 para leer cada página; para otros formatos asume
    codificación UTF-8.

    Args:
        file: Objeto UploadFile de FastAPI con el archivo subido.

    Returns:
        Cadena con el texto completo extraído del archivo.

    Raises:
        HTTPException: Si ocurre un error al procesar el PDF (400) o al
            leer el archivo de texto (400).
    """
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
    """Crea una huella digital del documento para detección de duplicados.

    Genera un identificador basado en los primeros 128 y últimos 128
    caracteres del texto. Esto permite detectar documentos duplicados
    independientemente del nombre de archivo.

    Args:
        text: Texto completo del documento.

    Returns:
        Cadena que sirve como huella digital del documento. Para textos
        de 256 caracteres o menos, devuelve el texto completo.
    """
    text = text.strip()
    if len(text) <= 256:
        # For very short documents, use the entire text
        return text

    # Use first 128 and last 128 characters
    fingerprint = text[:128] + text[-128:]
    return fingerprint


def remove_document_from_global_index(doc_id: int):
    """Elimina todos los fragmentos de un documento específico del índice global.

    Reconstruye el índice FAISS global excluyendo los embeddings del
    documento indicado y actualiza los metadatos correspondientes.

    Args:
        doc_id: Identificador numérico del documento a eliminar.
    """
    global global_faiss_index, global_chunk_metadata

    if not global_chunk_metadata:
        return

    # Find indices of chunks belonging to this document
    indices_to_remove = []
    new_metadata = []

    for i, meta in enumerate(global_chunk_metadata):
        if meta['doc_id'] != doc_id:
            new_metadata.append(meta)
        else:
            indices_to_remove.append(i)

    # If we have chunks to remove, rebuild the global index
    if indices_to_remove:
        global_chunk_metadata = new_metadata

        # Rebuild the global FAISS index without the removed document
        if new_metadata:
            # Get all remaining embeddings
            all_embeddings = []
            for meta in new_metadata:
                old_doc_id = meta['doc_id']
                chunk_idx = meta['chunk_index']
                if old_doc_id in documents:
                    doc_embeddings = documents[old_doc_id]['embeddings']
                    if chunk_idx < len(doc_embeddings):
                        all_embeddings.append(doc_embeddings[chunk_idx])

            if all_embeddings:
                # Rebuild global index
                embeddings_array = np.array(all_embeddings)
                global_faiss_index = faiss.IndexFlatL2(embeddings_array.shape[1])
                global_faiss_index.add(embeddings_array)

                # Update global indices in metadata
                for i, meta in enumerate(global_chunk_metadata):
                    meta['global_index'] = i
        else:
            # No documents left, reset global index
            global_faiss_index = None


def handle_duplicate_document(fingerprint: str, new_doc_id: int, filename: str) -> Dict[str, Any]:
    """Gestiona la detección y reemplazo de documentos duplicados.

    Si la huella digital ya existe en el registro, elimina el documento
    anterior de todas las estructuras de almacenamiento y lo reemplaza
    con el nuevo documento.

    Args:
        fingerprint: Huella digital del documento nuevo.
        new_doc_id: Identificador asignado al documento nuevo.
        filename: Nombre del archivo del documento nuevo.

    Returns:
        Diccionario con información sobre la acción realizada, incluyendo
        si se detectó un duplicado, la acción tomada ('replaced' o 'added'),
        y los identificadores relevantes.
    """
    if fingerprint in document_fingerprints:
        old_doc_id = document_fingerprints[fingerprint]
        old_filename = documents[old_doc_id]['filename'] if old_doc_id in documents else "Unknown"

        # Remove old document from all structures
        if old_doc_id in documents:
            remove_document_from_global_index(old_doc_id)
            del documents[old_doc_id]

        if old_doc_id in faiss_indexes:
            del faiss_indexes[old_doc_id]

        # Update fingerprint mapping to new document
        document_fingerprints[fingerprint] = new_doc_id

        return {
            "duplicate_detected": True,
            "action": "replaced",
            "old_doc_id": old_doc_id,
            "old_filename": old_filename,
            "new_doc_id": new_doc_id,
            "new_filename": filename
        }
    else:
        # New unique document
        document_fingerprints[fingerprint] = new_doc_id
        return {
            "duplicate_detected": False,
            "action": "added",
            "new_doc_id": new_doc_id,
            "new_filename": filename
        }


def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Divide el texto en fragmentos (chunks) con solapamiento.

    Segmenta el texto en fragmentos del tamaño indicado con un 10%% de
    solapamiento entre fragmentos consecutivos para mantener mejor
    continuidad de contexto.

    Args:
        text: Texto completo a segmentar.
        chunk_size: Tamaño máximo de cada fragmento en caracteres.
            Por defecto 1000, optimizado para procesamiento con vLLM.

    Returns:
        Lista de cadenas, cada una representando un fragmento de texto.
        Devuelve lista vacía si el texto está vacío.
    """
    text = text.strip()
    if not text:
        return []

    # Create chunks with slight overlap for better context
    chunks = []
    overlap = chunk_size // 10  # 10% overlap

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def create_embeddings(chunks: List[str]) -> np.ndarray:
    """Genera embeddings para los fragmentos de texto usando SentenceTransformer.

    Args:
        chunks: Lista de fragmentos de texto a codificar.

    Returns:
        Array numpy de forma (n_chunks, dimension) con los embeddings
        generados por el modelo all-MiniLM-L6-v2.
    """
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    return embeddings


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Construye un índice FAISS a partir de embeddings.

    Crea un índice de búsqueda por distancia euclidiana (L2) e inserta
    los embeddings proporcionados.

    Args:
        embeddings: Array numpy con los embeddings a indexar.

    Returns:
        Índice FAISS IndexFlatL2 listo para búsquedas.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def initialize_global_index(dimension: int):
    """Inicializa el índice FAISS global si aún no existe.

    Args:
        dimension: Dimensionalidad de los vectores de embeddings.
    """
    global global_faiss_index
    if global_faiss_index is None:
        global_faiss_index = faiss.IndexFlatL2(dimension)


def add_to_global_index(embeddings: np.ndarray, doc_id: int, filename: str, chunks: List[str]):
    """Añade los embeddings de un documento al índice global con metadatos.

    Inserta los vectores en el índice FAISS global y registra los metadatos
    de cada fragmento (documento de origen, índice del fragmento, texto)
    para la posterior atribución de fuentes en las búsquedas.

    Args:
        embeddings: Array numpy con los embeddings del documento.
        doc_id: Identificador numérico del documento.
        filename: Nombre del archivo del documento.
        chunks: Lista de fragmentos de texto correspondientes a los embeddings.
    """
    global global_faiss_index, global_chunk_metadata

    # Initialize global index if needed
    if global_faiss_index is None:
        initialize_global_index(embeddings.shape[1])

    # Add embeddings to global index
    global_faiss_index.add(embeddings)

    # Add metadata for each chunk
    for i, chunk in enumerate(chunks):
        global_chunk_metadata.append({
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "chunk_text": chunk,
            "global_index": len(global_chunk_metadata)  # Position in global index
        })


def search_global_index(query_embedding: np.ndarray, k: int = 10) -> tuple:
    """Busca en el índice FAISS global y devuelve resultados con metadatos.

    Realiza una búsqueda de los k vecinos más cercanos en el índice global
    que contiene fragmentos de todos los documentos cargados.

    Args:
        query_embedding: Array numpy con el embedding de la consulta,
            de forma (1, dimension).
        k: Número máximo de resultados a devolver. Por defecto 10.

    Returns:
        Tupla con dos listas:
            - Lista de cadenas con el texto de los fragmentos recuperados.
            - Lista de diccionarios con los metadatos de cada fragmento
              (doc_id, filename, chunk_index, chunk_text, global_index).
    """
    global global_faiss_index, global_chunk_metadata

    if global_faiss_index is None or len(global_chunk_metadata) == 0:
        return [], []

    # Search global index
    distances, indices = global_faiss_index.search(query_embedding, k)

    # Get metadata for retrieved chunks
    retrieved_metadata = []
    retrieved_chunks = []

    for idx in indices[0]:
        if 0 <= idx < len(global_chunk_metadata):
            metadata = global_chunk_metadata[idx]
            retrieved_metadata.append(metadata)
            retrieved_chunks.append(metadata["chunk_text"])

    return retrieved_chunks, retrieved_metadata


def summarize_with_vllm(text: str) -> str:
    """Genera un resumen del texto usando vLLM con prompts especializados.

    Para textos cortos (menos de 2000 caracteres) realiza un resumen directo.
    Para textos más largos, divide en fragmentos de hasta 3000 caracteres,
    resume cada uno individualmente y luego combina los resúmenes parciales
    en un resumen final comprehensivo.

    Args:
        text: Texto completo del documento a resumir.

    Returns:
        Cadena con el resumen generado por vLLM.

    Raises:
        HTTPException: Si ocurre un error durante la generación del resumen (500).
    """
    system_prompt = """You are an expert document summarizer. Create concise, accurate summaries that capture the main points and key insights. 
Focus on the most important information while maintaining clarity and coherence."""

    # Configure vLLM for summarization
    vllm_client.max_tokens = max(getattr(vllm_client, 'max_tokens', 0) or 0, 6000)
    vllm_client.temperature = 0.3  # Lower temperature for more focused summaries

    try:
        if len(text) < 2000:
            # Direct summarization for shorter texts
            prompt = f"""Summarize this document in 3-4 sentences, highlighting the main points:

{text}

Summary:"""
            return vllm_client.generate(prompt, system_instruction=system_prompt)
        else:
            # For longer texts, summarize by chunks then combine
            chunks = chunk_text(text, chunk_size=3000)  # Larger chunks for vLLM
            chunk_summaries = []

            for i, chunk in enumerate(chunks[:8]):  # Limit to 8 chunks for performance
                chunk_prompt = f"""Summarize this section (part {i + 1}) in 2-3 sentences:

{chunk}

Section summary:"""
                summary = vllm_client.generate(chunk_prompt, system_instruction=system_prompt)
                if summary.strip():
                    chunk_summaries.append(summary.strip())

            if not chunk_summaries:
                raise Exception("No chunk summaries produced.")

            # Combine all section summaries
            combined_summaries = "\n\n".join(chunk_summaries)
            final_prompt = f"""Create a comprehensive final summary from these section summaries:

{combined_summaries}

Final comprehensive summary:"""

            vllm_client.max_tokens = max(getattr(vllm_client, 'max_tokens', 0) or 0, 6000)  # Ensure at least 6000
            return vllm_client.generate(final_prompt, system_instruction=system_prompt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM summarization error: {str(e)}")


def answer_with_vllm(query: str, context_chunks: List[str]) -> str:
    """Responde preguntas usando vLLM con fragmentos de contexto.

    Construye un prompt con los fragmentos de contexto más relevantes
    y genera una respuesta fundamentada en la información proporcionada.

    Args:
        query: Pregunta del usuario.
        context_chunks: Lista de fragmentos de texto relevantes recuperados
            mediante búsqueda semántica.

    Returns:
        Cadena con la respuesta generada por vLLM.

    Raises:
        HTTPException: Si ocurre un error durante la generación de la respuesta (500).
    """
    system_prompt = """You are a helpful assistant that answers questions based on provided context. 
Be precise and informative. If the answer isn't clearly in the context, say so politely. 
Always base your response on the given information."""

    # Configure vLLM for Q&A
    vllm_client.max_tokens = max(getattr(vllm_client, 'max_tokens', 0) or 0, 6000)
    vllm_client.temperature = 0.2  # Lower temperature for factual accuracy

    try:
        # vLLM can handle more context than DistilBERT
        context = "\n\n".join(context_chunks[:8])  # Use more chunks

        prompt = f"""Context Information:
{context}

Question: {query}

Based on the context above, please provide a detailed answer:"""

        return vllm_client.generate(prompt, system_instruction=system_prompt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM Q&A error: {str(e)}")


def answer_global_with_vllm(query: str, context_chunks: List[str], metadata: List[Dict[str, Any]]) -> str:
    """Responde preguntas usando vLLM con contexto global y atribución de fuentes.

    Construye un prompt enriquecido con fragmentos de múltiples documentos,
    incluyendo la identificación del documento de origen de cada fragmento,
    para generar respuestas con atribución de fuentes.

    Args:
        query: Pregunta del usuario.
        context_chunks: Lista de fragmentos de texto relevantes de múltiples
            documentos.
        metadata: Lista de diccionarios con metadatos de cada fragmento,
            incluyendo 'filename' y 'doc_id'.

    Returns:
        Cadena con la respuesta generada por vLLM, incluyendo referencias
        a los documentos fuente cuando es pertinente.

    Raises:
        HTTPException: Si ocurre un error durante la generación de la respuesta (500).
    """
    system_prompt = """You are a helpful assistant that answers questions based on provided context from multiple documents. 
Be precise and informative. When possible, mention which document(s) support your answer. 
If the answer isn't clearly in the context, say so politely. Always base your response on the given information."""

    # Configure vLLM for global Q&A
    vllm_client.max_tokens = max(getattr(vllm_client, 'max_tokens', 0) or 0, 6000)  # Ensure at least 6000
    vllm_client.temperature = 0.2

    try:
        # Build context with source attribution
        context_with_sources = ""
        seen_docs = set()

        for i, (chunk, meta) in enumerate(zip(context_chunks[:8], metadata[:8])):
            doc_info = f"[Document: {meta['filename']}]"
            context_with_sources += f"{doc_info}\n{chunk}\n\n"
            seen_docs.add(meta['filename'])

        prompt = f"""Context Information from {len(seen_docs)} document(s):
{context_with_sources}

Question: {query}

Based on the context above from multiple documents, please provide a comprehensive answer. 
When relevant, mention which document(s) support specific points:"""

        return vllm_client.generate(prompt, system_instruction=system_prompt)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vLLM Global Q&A error: {str(e)}")


# ------------------------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Modelo de solicitud para consultas de preguntas y respuestas.

    Attributes:
        query: Texto de la pregunta del usuario.
    """

    query: str


class UploadResponse(BaseModel):
    """Modelo de respuesta para la carga de documentos.

    Attributes:
        doc_id: Identificador numérico asignado al documento.
        message: Mensaje descriptivo del resultado de la operación.
        duplicate_info: Información sobre la detección de duplicados,
            incluyendo si se reemplazó un documento previo. Puede ser None.
    """

    doc_id: int
    message: str
    duplicate_info: Dict[str, Any] = None  # Information about duplicate handling


class SummaryResponse(BaseModel):
    """Modelo de respuesta para resúmenes de documentos.

    Attributes:
        doc_id: Identificador del documento resumido.
        summary: Texto del resumen generado por vLLM.
    """

    doc_id: int
    summary: str


class QueryResponse(BaseModel):
    """Modelo de respuesta para consultas sobre un documento individual.

    Attributes:
        doc_id: Identificador del documento consultado.
        query: Texto de la pregunta original.
        answer: Respuesta generada por vLLM.
        context_chunks: Fragmentos de contexto utilizados para generar
            la respuesta.
    """

    doc_id: int
    query: str
    answer: str
    context_chunks: List[str]


class GlobalQueryResponse(BaseModel):
    """Modelo de respuesta para consultas globales sobre todos los documentos.

    Attributes:
        query: Texto de la pregunta original.
        answer: Respuesta generada por vLLM con atribución de fuentes.
        context_chunks: Fragmentos de contexto de múltiples documentos.
        sources: Lista de diccionarios con metadatos de los documentos
            fuente (doc_id, filename, chunks_used).
        total_documents_searched: Número de documentos distintos consultados.
    """

    query: str
    answer: str
    context_chunks: List[str]
    sources: List[Dict[str, Any]]  # Document sources with metadata
    total_documents_searched: int


class DocumentInfo(BaseModel):
    """Modelo con información básica de un documento cargado.

    Attributes:
        doc_id: Identificador numérico del documento.
        filename: Nombre original del archivo subido.
        text_length: Longitud total del texto extraído en caracteres.
        num_chunks: Número de fragmentos generados tras la segmentación.
    """

    doc_id: int
    filename: str
    text_length: int
    num_chunks: int


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Carga y procesa un documento usando vLLM para capacidades de IA mejoradas.

    Gestiona documentos duplicados reemplazando versiones anteriores con las nuevas.

    Proceso:
        1. Extrae texto del archivo PDF o de texto plano.
        2. Genera huella digital del documento para detección de duplicados.
        3. Gestiona duplicados (reemplaza la versión anterior si se encuentra).
        4. Divide el texto en fragmentos optimizados para procesamiento con vLLM.
        5. Genera embeddings para búsqueda semántica.
        6. Crea índice FAISS para recuperación rápida.
        7. Actualiza el índice global.

    Args:
        file: Archivo subido por el usuario (PDF o texto plano).

    Returns:
        UploadResponse con el ID del documento, mensaje de estado e
        información sobre duplicados.

    Raises:
        HTTPException: Si el archivo no contiene texto extraíble (400) o
            si el procesamiento no produce contenido (400).
    """
    global doc_id_counter

    text = extract_text(file)
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text found.")

    # Generate document fingerprint for duplicate detection
    fingerprint = create_document_fingerprint(text)

    # Check for duplicates and handle them
    doc_id = doc_id_counter
    duplicate_info = handle_duplicate_document(fingerprint, doc_id, file.filename)

    # Create chunks optimized for vLLM (larger chunks)
    chunks = chunk_text(text, chunk_size=1000)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document processing resulted in no content.")

    # Generate embeddings for semantic search
    embeddings = create_embeddings(chunks)

    # Build FAISS index
    index = create_faiss_index(embeddings)

    # Store document
    documents[doc_id] = {
        "filename": file.filename,
        "text": text,
        "chunks": chunks,
        "embeddings": embeddings,
        "fingerprint": fingerprint,
        "upload_timestamp": __import__('datetime').datetime.now().isoformat()
    }
    faiss_indexes[doc_id] = index

    # Add to global index
    add_to_global_index(embeddings, doc_id, file.filename, chunks)

    # Only increment counter if it's a new document (not a replacement)
    if not duplicate_info["duplicate_detected"]:
        doc_id_counter += 1

    # Prepare response message
    if duplicate_info["duplicate_detected"]:
        message = f"Document processed successfully. Replaced previous version (ID: {duplicate_info['old_doc_id']}) with new version. Model: {vllm_client.model_name}"
    else:
        message = f"Document uploaded and processed successfully using vLLM backend. Model: {vllm_client.model_name}"

    return UploadResponse(
        doc_id=doc_id,
        message=message,
        duplicate_info=duplicate_info
    )


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """Lista todos los documentos cargados con sus metadatos.

    Returns:
        Lista de objetos DocumentInfo con información básica de cada
        documento almacenado en memoria.
    """
    docs_info = []
    for doc_id, data in documents.items():
        docs_info.append(DocumentInfo(
            doc_id=doc_id,
            filename=data.get("filename", "Unknown"),
            text_length=len(data.get("text", "")),
            num_chunks=len(data.get("chunks", []))
        ))
    return docs_info


@app.get("/document/{doc_id}", response_model=DocumentInfo)
async def get_document_info(doc_id: int):
    """Obtiene información detallada de un documento específico.

    Args:
        doc_id: Identificador numérico del documento.

    Returns:
        Objeto DocumentInfo con los metadatos del documento.

    Raises:
        HTTPException: Si el documento no se encuentra (404).
    """
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    data = documents[doc_id]
    return DocumentInfo(
        doc_id=doc_id,
        filename=data.get("filename", "Unknown"),
        text_length=len(data.get("text", "")),
        num_chunks=len(data.get("chunks", []))
    )


@app.get("/document/{doc_id}/summary", response_model=SummaryResponse)
async def summarize_document(doc_id: int):
    """Genera un resumen inteligente del documento usando vLLM.

    Utiliza el modelo configurado en vLLM para resúmenes de alta calidad con:
        - Segmentación consciente del contexto para documentos largos.
        - Prompts especializados según la longitud del documento.
        - Optimización de temperatura para resúmenes coherentes.

    Args:
        doc_id: Identificador numérico del documento a resumir.

    Returns:
        SummaryResponse con el ID del documento y el resumen generado.

    Raises:
        HTTPException: Si el documento no se encuentra (404) o si ocurre
            un error al generar el resumen (500).
    """
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    text = documents[doc_id]["text"]

    try:
        summary = summarize_with_vllm(text)
        return SummaryResponse(doc_id=doc_id, summary=summary)
    except Exception as e:
        # Log the error but return a user-friendly message
        print(f"Summarization error for doc {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating summary. Please try again.")


@app.post("/document/{doc_id}/query", response_model=QueryResponse)
async def query_document(doc_id: int, query_request: QueryRequest):
    """Responde preguntas sobre un documento usando vLLM con búsqueda semántica.

    Proceso:
        1. Genera embedding para la consulta.
        2. Encuentra los fragmentos más relevantes del documento usando FAISS.
        3. Usa vLLM con ventana de contexto ampliada para respuestas inteligentes.
        4. Devuelve la respuesta junto con los fragmentos fuente para transparencia.

    Args:
        doc_id: Identificador numérico del documento a consultar.
        query_request: Objeto con la pregunta del usuario.

    Returns:
        QueryResponse con la pregunta, respuesta y fragmentos de contexto.

    Raises:
        HTTPException: Si el documento no se encuentra (404), si no se
            pueden recuperar fragmentos relevantes (500), o si ocurre
            un error al generar la respuesta (500).
    """
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    query = query_request.query

    # Generate query embedding
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # Retrieve relevant chunks using FAISS
    index = faiss_indexes[doc_id]
    k = 6  # More chunks for vLLM (vs 3 for DistilBERT)
    distances, indices = index.search(query_embedding, k)

    # Get relevant text chunks
    chunks = documents[doc_id]["chunks"]
    retrieved_chunks = []

    for idx in indices[0]:
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])

    if not retrieved_chunks:
        raise HTTPException(status_code=500, detail="Unable to retrieve relevant context for the query.")

    # Generate answer using vLLM
    try:
        answer = answer_with_vllm(query, retrieved_chunks)

        return QueryResponse(
            doc_id=doc_id,
            query=query,
            answer=answer,
            context_chunks=retrieved_chunks
        )
    except Exception as e:
        print(f"Q&A error for doc {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Error generating answer. Please try again.")


@app.post("/query_all", response_model=GlobalQueryResponse)
async def query_all_documents(query_request: QueryRequest):
    """Busca y responde preguntas a través de TODOS los documentos de la base.

    Proceso:
        1. Genera embedding para la consulta.
        2. Busca en el índice FAISS global a través de todos los documentos.
        3. Recupera los top-k fragmentos de múltiples documentos.
        4. Usa vLLM para generar una respuesta con atribución de fuentes.
        5. Devuelve la respuesta con información de los documentos fuente.

    Args:
        query_request: Objeto con la pregunta del usuario.

    Returns:
        GlobalQueryResponse con la respuesta, fragmentos de contexto,
        fuentes y cantidad total de documentos consultados.

    Raises:
        HTTPException: Si no hay documentos en la base (404) o si ocurre
            un error al generar la respuesta (500).
    """
    if not global_chunk_metadata:
        raise HTTPException(status_code=404, detail="No documents found in the database.")

    query = query_request.query

    # Generate query embedding
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)

    # Search global index
    k = 10  # More chunks for global search
    retrieved_chunks, metadata = search_global_index(query_embedding, k)

    if not retrieved_chunks:
        raise HTTPException(status_code=500, detail="Unable to retrieve relevant context from any document.")

    # Generate answer using vLLM with global context
    try:
        answer = answer_global_with_vllm(query, retrieved_chunks, metadata)

        # Build sources information
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
        print(f"Global Q&A error: {e}")
        raise HTTPException(status_code=500, detail="Error generating global answer. Please try again.")


@app.get("/document/{doc_id}/chunks")
async def get_document_chunks(doc_id: int):
    """Recupera los fragmentos de texto procesados de un documento.

    Útil para depuración o interfaces avanzadas que necesiten visualizar
    los fragmentos individuales.

    Args:
        doc_id: Identificador numérico del documento.

    Returns:
        Diccionario con doc_id, lista de chunks y total_chunks.

    Raises:
        HTTPException: Si el documento no se encuentra (404).
    """
    if doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "doc_id": doc_id,
        "chunks": documents[doc_id]["chunks"],
        "total_chunks": len(documents[doc_id]["chunks"])
    }


@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud del sistema con estado de vLLM.

    Comprueba la conexión con el servidor vLLM y devuelve información
    sobre el estado general del sistema, incluyendo documentos cargados,
    estado del índice global y configuración de detección de duplicados.

    Returns:
        Diccionario con el estado del sistema, configuración de vLLM,
        estadísticas de documentos e índice global.
    """
    try:
        # Test vLLM connection
        test_response = vllm_client.generate("Hello", system_instruction="Respond with 'OK'")
        vllm_status = "connected" if test_response else "error"
    except Exception as e:
        vllm_status = f"error: {str(e)}"

    global_chunks = len(global_chunk_metadata) if global_chunk_metadata else 0
    unique_docs_in_global = len(set(meta['doc_id'] for meta in global_chunk_metadata)) if global_chunk_metadata else 0

    return {
        "status": "healthy",
        "vllm_server": vllm_client.base_url,
        "vllm_model": vllm_client.model_name,
        "vllm_status": vllm_status,
        "total_documents": len(documents),
        "total_fingerprints": len(document_fingerprints),
        "embedding_model": "all-MiniLM-L6-v2",
        "global_index": {
            "chunks": global_chunks,
            "documents": unique_docs_in_global,
            "status": "active" if global_faiss_index is not None else "inactive"
        },
        "duplicate_detection": {
            "enabled": True,
            "method": "first_128_last_128_chars",
            "policy": "replace_with_newest"
        }
    }


@app.get("/duplicates")
async def get_duplicate_info():
    """Obtiene información sobre las huellas digitales y detección de duplicados.

    Devuelve un listado de todas las huellas digitales registradas junto
    con los datos del documento asociado a cada una.

    Returns:
        Diccionario con el total de documentos únicos, método de detección,
        política de reemplazo y lista detallada de documentos.
    """
    fingerprint_info = []

    for fingerprint, doc_id in document_fingerprints.items():
        if doc_id in documents:
            doc = documents[doc_id]
            fingerprint_info.append({
                "fingerprint": fingerprint[:50] + "..." if len(fingerprint) > 50 else fingerprint,
                "doc_id": doc_id,
                "filename": doc['filename'],
                "upload_timestamp": doc.get('upload_timestamp', 'Unknown'),
                "text_length": len(doc['text']),
                "chunks_count": len(doc['chunks'])
            })

    return {
        "total_unique_documents": len(document_fingerprints),
        "duplicate_detection_method": "first_128_last_128_characters",
        "replacement_policy": "newest_version_predominates",
        "documents": fingerprint_info
    }


@app.get("/config")
async def get_configuration():
    """Obtiene la configuración actual del sistema.

    Devuelve los parámetros de configuración de vLLM, modelo de embeddings,
    tamaño de fragmentos, estado del índice global y configuración de
    detección de duplicados.

    Returns:
        Diccionario con toda la configuración del sistema.
    """
    global_index_size = len(global_chunk_metadata) if global_chunk_metadata else 0

    return {
        "vllm_config": {
            "base_url": vllm_client.base_url,
            "model": vllm_client.model_name,
            "max_tokens": vllm_client.max_tokens,
            "temperature": vllm_client.temperature
        },
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimensions": 384,
        "chunk_size": 1000,
        "max_chunks_for_qa": 6,
        "global_index": {
            "enabled": True,
            "total_chunks": global_index_size,
            "total_documents": len(
                set(meta['doc_id'] for meta in global_chunk_metadata)) if global_chunk_metadata else 0
        },
        "duplicate_detection": {
            "enabled": True,
            "fingerprint_method": "first_128_last_128_chars",
            "policy": "replace_with_newest",
            "total_unique_fingerprints": len(document_fingerprints)
        }
    }


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting SmartDoc Backend with vLLM integration...")
    print(f"📡 vLLM Server: {VLLM_BASE_URL}")
    print(f"🤖 vLLM Model: {VLLM_MODEL}")
    print("🔍 Embedding Model: all-MiniLM-L6-v2 (local)")
    print("🌐 Global Index: Enabled for cross-document search")
    print("🔄 Duplicate Detection: Enabled (newest version predominates)")
    print("👆 Fingerprint Method: First 128 + Last 128 characters")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8001)
