"""
Modelos generadores de lenguaje (LLM) con soporte para vLLM, Gemini y LangChain.

Módulo que proporciona una jerarquía de clases para interactuar con distintos
motores de inferencia de modelos de lenguaje grande (LLM). Define una interfaz
abstracta base (SimpleLLM) e implementaciones concretas para:

    - GeminiLLM: Cliente directo para la API de Google Gemini.
    - VLLMClient: Cliente síncrono/asíncrono para servidores vLLM con soporte
      para generación por lotes (batch), streaming y control de concurrencia.
    - LangChainVLLMAdapter: Adaptador que integra vLLM a través de LangChain
      con fallback automático a Gemini en caso de fallo.

Todas las implementaciones fuerzan por defecto la respuesta en español mediante
un mensaje de sistema predeterminado (DEFAULT_SYSTEM_ES).
"""

import os
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Dict, Any
import concurrent.futures
from contextlib import asynccontextmanager

from google import genai
from google.genai import types
from openai import OpenAI, AsyncOpenAI
from langchain_openai import ChatOpenAI as LCChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = 'AIzaSyCS6eXsbMFuwz8BPMn0q5FWpV4yYCCejC0'
GEMINI_MODEL = "gemini-2.0-flash-lite"

# vLLM Configuration
VLLM_BASE_URL = "http://172.24.250.17:8000/v1"
VLLM_MODEL = "Qwen3-8B-AWQ"
VLLM_API_KEY = "EMPTY"  # vLLM doesn't require a real API key

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', GEMINI_API_KEY)
GEMINI_MODEL = os.getenv('GEMINI_MODEL', GEMINI_MODEL)
VLLM_BASE_URL = os.getenv('VLLM_BASE_URL', VLLM_BASE_URL)
VLLM_MODEL = os.getenv('VLLM_MODEL', VLLM_MODEL)
USE_GEMINI_DIRECT = os.getenv('USE_GEMINI', '0').strip().lower() in {'1', 'true', 'yes'}

# Enforce Spanish by default
DEFAULT_SYSTEM_ES = (
    "Eres un asistente útil. Responde SIEMPRE en español. "
    "Si el usuario escribe en otro idioma, traduce su intención y responde en español."
)


# ================================
# INTERFACE BASE
# ================================

class SimpleLLM(ABC):
    """Interfaz abstracta base para modelos de lenguaje grande (LLM).

    Define el contrato común que deben implementar todos los clientes LLM
    del sistema, incluyendo generación síncrona, chat y sus variantes con
    streaming.

    Attributes:
        _max_tokens: Número máximo de tokens a generar en cada respuesta.
        _temperature: Parámetro de temperatura que controla la aleatoriedad
            de la generación (0.0 = determinista, 2.0 = máxima aleatoriedad).
    """

    def __init__(self, max_tokens: int = 150, temperature: float = 0.7):
        self._max_tokens = max_tokens
        self._temperature = temperature

    @property
    def max_tokens(self) -> int:
        """Obtiene el número máximo de tokens configurado."""
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        """Establece el número máximo de tokens.

        Args:
            value: Entero positivo con el nuevo límite de tokens.

        Raises:
            TypeError: Si el valor no es un entero.
            ValueError: Si el valor es menor o igual a 0.
        """
        if not isinstance(value, int):
            raise TypeError("max_tokens must be an integer")
        if value <= 0:
            raise ValueError("max_tokens must be greater than 0")
        self._max_tokens = value

    @property
    def temperature(self) -> float:
        """Obtiene el valor de temperatura configurado."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Establece el valor de temperatura para la generación.

        Args:
            value: Número entre 0.0 y 2.0.

        Raises:
            TypeError: Si el valor no es numérico.
            ValueError: Si el valor está fuera del rango [0.0, 2.0].
        """
        if not isinstance(value, (int, float)):
            raise TypeError("temperature must be a number")
        if not 0.0 <= value <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        self._temperature = float(value)

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Genera una respuesta a partir de un prompt de texto.

        Args:
            prompt: Texto de entrada para la generación.

        Returns:
            Cadena con la respuesta generada por el modelo.
        """
        pass

    @abstractmethod
    def chat(self, messages: list) -> str:
        """Genera una respuesta a partir de un historial de mensajes.

        Args:
            messages: Lista de diccionarios con claves 'role' y 'content'.

        Returns:
            Cadena con la respuesta generada por el modelo.
        """
        pass

    @abstractmethod
    def generate_stream(self, prompt: str) -> Iterator:
        """Genera una respuesta en streaming a partir de un prompt.

        Args:
            prompt: Texto de entrada para la generación.

        Returns:
            Iterador que produce fragmentos de la respuesta.
        """
        pass

    @abstractmethod
    def chat_stream(self, messages: list) -> Iterator:
        """Genera una respuesta en streaming a partir de un historial de mensajes.

        Args:
            messages: Lista de diccionarios con claves 'role' y 'content'.

        Returns:
            Iterador que produce fragmentos de la respuesta.
        """
        pass


class GeminiLLM(SimpleLLM):
    """Cliente directo para la API de Google Gemini.

    Implementa la interfaz SimpleLLM para interactuar con modelos Gemini,
    soportando generación síncrona, chat y streaming. Formatea las
    conversaciones con roles personalizados (CLIENTE / MARIA JESUS).

    Attributes:
        api_key: Clave de API de Google Gemini.
        client: Instancia del cliente genai de Google.
        model_name: Nombre del modelo Gemini a utilizar.
    """

    def __init__(self, api_key: str = None, model_name: str = None,
                 max_tokens: int = 6000, temperature: float = 0.7):
        super().__init__(max_tokens, temperature)
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY must be provided or set in environment")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name or GEMINI_MODEL

    def _get_config(self, system_instruction: str = None) -> types.GenerateContentConfig:
        """Construye la configuración de generación de contenido para Gemini.

        Args:
            system_instruction: Instrucción de sistema personalizada. Si es None,
                utiliza DEFAULT_SYSTEM_ES para forzar respuestas en español.

        Returns:
            Objeto GenerateContentConfig con los parámetros configurados.
        """
        sys_inst = system_instruction or DEFAULT_SYSTEM_ES
        return types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            system_instruction=sys_inst,
        )

    def generate_stream(self, prompt: str) -> Iterator:
        """Genera una respuesta en streaming a partir de un prompt.

        Args:
            prompt: Texto de entrada para la generación.

        Returns:
            Iterador de streaming con los fragmentos de la respuesta de Gemini.
        """
        config = self._get_config()
        response_stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return response_stream

    def chat_stream(self, messages: list = []) -> Iterator:
        """Genera una respuesta de chat en streaming a partir de un historial.

        Formatea los mensajes con roles CLIENTE y MARIA JESUS, y utiliza
        los últimos 6 mensajes del historial para mantener contexto.

        Args:
            messages: Lista de diccionarios con 'role' y 'content'.
                Si el primer mensaje es 'system', se usa como instrucción.

        Returns:
            Iterador de streaming con los fragmentos de la respuesta.
        """
        system_prompt: str = DEFAULT_SYSTEM_ES
        if len(messages) and messages[0]['role'] == 'system':
            system_prompt = messages[0]['content'] or DEFAULT_SYSTEM_ES

        prompt = ""
        for msg in messages[-6:]:
            role_name = "CLIENTE" if msg['role'] == 'user' else "MARÍA JESÚS"
            prompt += f"{role_name}: {msg['content']}\n\n"
        prompt += f"\n\nMARÍA JESÚS:"

        config = self._get_config(system_instruction=system_prompt)
        response_stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return response_stream

    def generate(self, prompt: str) -> str:
        """Genera una respuesta completa a partir de un prompt.

        Args:
            prompt: Texto de entrada para la generación.

        Returns:
            Cadena con el texto de la respuesta completa de Gemini.
        """
        config = self._get_config()
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return response.text

    def chat(self, messages: list = []) -> str:
        """Genera una respuesta de chat a partir de un historial de mensajes.

        Formatea los mensajes con roles CLIENTE y MARIA JESUS, y utiliza
        los últimos 6 mensajes del historial para mantener contexto.

        Args:
            messages: Lista de diccionarios con 'role' y 'content'.
                Si el primer mensaje es 'system', se usa como instrucción.

        Returns:
            Cadena con la respuesta completa generada por Gemini.
        """
        system_prompt: str = DEFAULT_SYSTEM_ES
        if len(messages) and messages[0]['role'] == 'system':
            system_prompt = messages[0]['content'] or DEFAULT_SYSTEM_ES

        prompt = ""
        for msg in messages[-6:]:
            role_name = "CLIENTE" if msg['role'] == 'user' else "MARÍA JESÚS"
            prompt += f"{role_name}: {msg['content']}\n\n"
        prompt += f"\n\nMARÍA JESÚS:"

        config = self._get_config(system_instruction=system_prompt)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )
        return response.text


# ================================
# vLLM IMPLEMENTATION (NO SINGLETON)
# ================================

class VLLMClient(SimpleLLM):
    """Cliente vLLM con soporte síncrono, asíncrono y procesamiento por lotes.

    Implementa la interfaz SimpleLLM para comunicarse con un servidor vLLM
    a través de la API compatible con OpenAI. Soporta generación síncrona,
    asíncrona, streaming y procesamiento por lotes con control de concurrencia
    mediante semáforos.

    No implementa patrón Singleton; cada instancia mantiene sus propios
    clientes sync y async.

    Attributes:
        base_url: URL base del servidor vLLM.
        model_name: Nombre del modelo desplegado en vLLM.
        api_key: Clave de API (normalmente 'EMPTY' para vLLM).
        max_concurrent_requests: Límite de solicitudes concurrentes para
            procesamiento por lotes.
        request_timeout: Tiempo máximo de espera por solicitud en segundos.
        client: Cliente OpenAI síncrono.
        async_client: Cliente AsyncOpenAI para operaciones asíncronas.
    """

    def __init__(self, base_url: str = None, model_name: str = None,
                 api_key: str = None, max_tokens: int = 10_000, temperature: float = 0.7,
                 max_concurrent_requests: int = 32, request_timeout: int = 300):
        super().__init__(max_tokens, temperature)

        # Set up vLLM client configuration
        self.base_url = base_url or VLLM_BASE_URL
        self.model_name = model_name or VLLM_MODEL
        self.api_key = api_key or VLLM_API_KEY
        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout

        # Create sync OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout
        )

        # Create async OpenAI client for batch processing
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout
        )

        # Semaphore for rate limiting concurrent requests
        self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)

    def _prepare_messages(self, messages: list) -> list:
        """Convierte mensajes al formato OpenAI y asegura mensaje de sistema en español.

        Filtra mensajes con roles válidos ('system', 'user', 'assistant') y
        añade el mensaje de sistema en español si no existe uno.

        Args:
            messages: Lista de diccionarios con 'role' y 'content'.

        Returns:
            Lista de diccionarios en formato compatible con la API OpenAI.
        """
        openai_messages = []
        has_system = False
        for msg in messages:
            if msg['role'] in ['system', 'user', 'assistant']:
                if msg['role'] == 'system':
                    has_system = True
                openai_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        if not has_system:
            openai_messages.insert(0, {'role': 'system', 'content': DEFAULT_SYSTEM_ES})
        return openai_messages

    def _prepare_prompt_as_messages(self, prompt: str, system_instruction: str = None) -> list:
        """Convierte un prompt simple al formato de mensajes de OpenAI.

        Crea una lista de mensajes con instrucción de sistema (personalizada
        o DEFAULT_SYSTEM_ES) y el prompt del usuario.

        Args:
            prompt: Texto del prompt del usuario.
            system_instruction: Instrucción de sistema personalizada. Si es
                None, utiliza DEFAULT_SYSTEM_ES.

        Returns:
            Lista de diccionarios en formato OpenAI con mensajes system y user.
        """
        messages = []
        if system_instruction:
            messages.append({'role': 'system', 'content': system_instruction})
        else:
            messages.append({'role': 'system', 'content': DEFAULT_SYSTEM_ES})
        messages.append({'role': 'user', 'content': prompt})
        return messages

    def _strip_think(self, text: str) -> str:
        """Elimina las etiquetas de pensamiento (<think>) de la respuesta del modelo.

        Algunos modelos (como Qwen) envuelven su razonamiento interno en
        etiquetas <think>...</think>. Este método extrae solo el contenido
        final posterior a dichas etiquetas.

        Args:
            text: Texto de la respuesta del modelo.

        Returns:
            Texto limpio sin las etiquetas de pensamiento.
        """
        if not text:
            return text
        if '</think>' in text:
            return text.split('</think>')[-1].strip()
        return text

    def generate(self, prompt: str, system_instruction: str = None) -> str:
        """Genera una respuesta completa a partir de un prompt.

        Args:
            prompt: Texto de entrada para la generación.
            system_instruction: Instrucción de sistema opcional. Si es None,
                usa DEFAULT_SYSTEM_ES.

        Returns:
            Cadena con la respuesta generada, limpia de etiquetas <think>.
        """
        messages = self._prepare_prompt_as_messages(prompt, system_instruction)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )

            res = response.choices[0].message.content
            return self._strip_think(res)

        except Exception as e:
            logger.error(f"Error in generate: {e}")
            return f"Error: {str(e)}"

    def chat(self, messages: list = []) -> str:
        """Genera una respuesta a partir de un historial de mensajes de chat.

        Args:
            messages: Lista de diccionarios con 'role' y 'content'.

        Returns:
            Cadena con la respuesta generada, o cadena vacía si no hay mensajes.
        """
        if not messages:
            return ""

        openai_messages = self._prepare_messages(messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            res = response.choices[0].message.content
            return self._strip_think(res)

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error: {str(e)}"

    # ================================
    # FIXED AND IMPROVED BATCH METHODS
    # ================================

    def batch_generate(self, batch_prompt: list, system_instruction: str = None) -> list:
        """Genera respuestas para un lote de prompts usando procesamiento asíncrono.

        Delega en batch_generate_async_wrapper para procesamiento paralelo.

        Args:
            batch_prompt: Lista de cadenas con los prompts a procesar.
            system_instruction: Instrucción de sistema opcional aplicada
                a todos los prompts del lote.

        Returns:
            Lista de cadenas con las respuestas generadas, en el mismo
            orden que los prompts de entrada.
        """
        return self.batch_generate_async_wrapper(batch_prompt, system_instruction)

    async def _async_generate_single(self, prompt: str, system_instruction: str = None) -> str:
        """Genera una respuesta individual de forma asíncrona con limitación de tasa.

        Utiliza un semáforo para controlar la concurrencia y evitar
        saturar el servidor vLLM.

        Args:
            prompt: Texto del prompt a procesar.
            system_instruction: Instrucción de sistema opcional.

        Returns:
            Cadena con la respuesta generada o mensaje de error.
        """
        async with self._semaphore:  # Rate limiting
            try:
                messages = self._prepare_prompt_as_messages(prompt, system_instruction)

                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=False
                )

                # Small delay to prevent overwhelming the server
                await asyncio.sleep(0.01)
                return self._strip_think(response.choices[0].message.content)

            except Exception as e:
                logger.error(f"Error in async generate for prompt: {prompt[:50]}..., Error: {e}")
                return f"Error: {str(e)}"

    async def batch_generate_async(self, prompts: List[str],
                                   system_instruction: str = None,
                                   chunk_size: int = None) -> List[str]:
        """Procesamiento asíncrono por lotes con segmentación y manejo de errores.

        Divide la lista de prompts en bloques (chunks) para procesarlos
        de forma paralela sin saturar el servidor, con pausas breves
        entre bloques.

        Args:
            prompts: Lista de cadenas con los prompts a procesar.
            system_instruction: Instrucción de sistema opcional aplicada
                a todos los prompts.
            chunk_size: Tamaño del bloque de procesamiento. Si es None,
                usa max_concurrent_requests.

        Returns:
            Lista de cadenas con las respuestas, en el mismo orden que
            los prompts de entrada. Los errores se devuelven como cadenas
            con prefijo 'Error:'.
        """
        if chunk_size is None:
            chunk_size = self.max_concurrent_requests

        all_results = []

        # Process prompts in chunks to avoid overwhelming the server
        for i in range(0, len(prompts), chunk_size):
            chunk = prompts[i:i + chunk_size]
            logger.info(
                f"Processing chunk {i // chunk_size + 1}/{(len(prompts) - 1) // chunk_size + 1} ({len(chunk)} prompts)")

            # Create tasks for this chunk
            tasks = [
                self._async_generate_single(prompt, system_instruction)
                for prompt in chunk
            ]

            # Process chunk with error handling
            try:
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Convert exceptions to error strings
                processed_results = []
                for result in chunk_results:
                    if isinstance(result, Exception):
                        processed_results.append(f"Error: {str(result)}")
                    else:
                        processed_results.append(result)

                all_results.extend(processed_results)

            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                # Add error for each prompt in the failed chunk
                all_results.extend([f"Chunk Error: {str(e)}"] * len(chunk))

            # Brief pause between chunks
            if i + chunk_size < len(prompts):
                await asyncio.sleep(0.1)

        return all_results

    def batch_generate_async_wrapper(self, prompts: List[str],
                                     system_instruction: str = None,
                                     chunk_size: int = None) -> List[str]:
        """Envoltorio síncrono para el procesamiento asíncrono por lotes.

        Crea o reutiliza un bucle de eventos asyncio para ejecutar
        batch_generate_async de forma síncrona.

        Args:
            prompts: Lista de cadenas con los prompts a procesar.
            system_instruction: Instrucción de sistema opcional.
            chunk_size: Tamaño del bloque de procesamiento.

        Returns:
            Lista de cadenas con las respuestas generadas.

        Raises:
            RuntimeError: Si se invoca dentro de un bucle de eventos
                asyncio ya en ejecución.
        """
        try:
            # Get or create event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to use asyncio.create_task
                # But since this is a sync wrapper, we'll use run_until_complete
                raise RuntimeError(
                    "Cannot run async wrapper in already running event loop. Use batch_generate_async directly.")
        except RuntimeError:
            # No event loop running, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.batch_generate_async(prompts, system_instruction, chunk_size)
            )
        finally:
            # Clean up the loop if we created it
            if not loop.is_running():
                loop.close()

    def batch_generate_threaded(self, prompts: List[str],
                                system_instruction: str = None,
                                max_workers: int = None) -> List[str]:
        """Procesamiento alternativo por lotes usando hilos (opción de respaldo).

        Utiliza ThreadPoolExecutor como alternativa al procesamiento
        asíncrono cuando no se puede usar asyncio.

        Args:
            prompts: Lista de cadenas con los prompts a procesar.
            system_instruction: Instrucción de sistema opcional.
            max_workers: Número máximo de hilos. Si es None, usa el
                mínimo entre max_concurrent_requests y la cantidad de prompts.

        Returns:
            Lista de cadenas con las respuestas generadas.
        """
        if max_workers is None:
            max_workers = min(self.max_concurrent_requests, len(prompts))

        def process_single(prompt):
            try:
                return self.generate(prompt, system_instruction)
            except Exception as e:
                logger.error(f"Thread error for prompt: {prompt[:50]}..., Error: {e}")
                return f"Error: {str(e)}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single, prompts))

        return results

    # ================================
    # STREAMING METHODS
    # ================================

    def generate_stream(self, prompt: str, system_instruction: str = None) -> Iterator:
        """Genera una respuesta en streaming a partir de un prompt.

        Args:
            prompt: Texto de entrada para la generación.
            system_instruction: Instrucción de sistema opcional.

        Returns:
            Iterador que produce fragmentos de la respuesta del servidor vLLM.
            Devuelve un iterador vacío si ocurre un error.
        """
        messages = self._prepare_prompt_as_messages(prompt, system_instruction)

        try:
            response_stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            return response_stream
        except Exception as e:
            logger.error(f"Error in generate_stream: {e}")
            return iter([])

    def chat_stream(self, messages: list = []) -> Iterator:
        """Genera una respuesta de chat en streaming a partir de un historial.

        Args:
            messages: Lista de diccionarios con 'role' y 'content'.

        Returns:
            Iterador que produce fragmentos de la respuesta del servidor vLLM.
            Devuelve un iterador vacío si no hay mensajes o si ocurre un error.
        """
        if not messages:
            return iter([])

        openai_messages = self._prepare_messages(messages)

        try:
            response_stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            return response_stream
        except Exception as e:
            logger.error(f"Error in chat_stream: {e}")
            return iter([])

    # ================================
    # UTILITY METHODS
    # ================================

    def health_check(self) -> Dict[str, Any]:
        """Verifica si el servidor vLLM está respondiendo correctamente.

        Envía una solicitud de prueba y evalúa la respuesta.

        Returns:
            Diccionario con el estado ('healthy' o 'unhealthy'), URL del
            servidor, modelo y respuesta de prueba o mensaje de error.
        """
        try:
            response = self.generate("Hello", "You are a helpful assistant.")
            return {
                "status": "healthy",
                "server_url": self.base_url,
                "model": self.model_name,
                "test_response": response[:100] + "..." if len(response) > 100 else response
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "server_url": self.base_url,
                "error": str(e)
            }

    def get_performance_settings(self) -> Dict[str, Any]:
        """Obtiene la configuración actual de rendimiento del cliente.

        Returns:
            Diccionario con max_concurrent_requests, request_timeout,
            max_tokens, temperature, base_url y model.
        """
        return {
            "max_concurrent_requests": self.max_concurrent_requests,
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "model": self.model_name
        }


# ================================
# LangChain adapter (obligatorio)
# ================================

class LangChainVLLMAdapter:
    """Adaptador de vLLM a través de LangChain con fallback a Gemini.

    Clase obligatoria en el proyecto que encapsula la comunicación con
    vLLM mediante LangChain (ChatOpenAI). Si el modo USE_GEMINI está
    activado o si vLLM falla, recurre automáticamente a GeminiLLM
    como mecanismo de respaldo.

    Attributes:
        base_url: URL base del servidor vLLM.
        model_name: Nombre del modelo en el servidor vLLM.
        api_key: Clave de API para vLLM.
        max_tokens: Número máximo de tokens por respuesta.
        temperature: Parámetro de temperatura para la generación.
        timeout: Tiempo máximo de espera por solicitud en segundos.
        llm: Instancia de LangChain ChatOpenAI (None si se usa Gemini directo).
        _prefer_gemini: Si True, usa Gemini como motor principal.
        _fallback_enabled: Si True, permite caer a Gemini ante fallos de vLLM.
        _fallback_llm: Instancia de GeminiLLM inicializada bajo demanda.
    """

    def __init__(self, base_url: str = None, model_name: str = None,
                 api_key: str = None, max_tokens: int = 10_000, temperature: float = 0.7,
                 timeout: int = 60):
        self._prefer_gemini = USE_GEMINI_DIRECT
        self.base_url = base_url or VLLM_BASE_URL
        self.model_name = model_name or VLLM_MODEL
        self.api_key = api_key or VLLM_API_KEY
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self.llm = None
        if not self._prefer_gemini:
            self.llm = LCChatOpenAI(
                model=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        self._fallback_enabled = not self._prefer_gemini
        self._fallback_llm: Optional[GeminiLLM] = None

    @staticmethod
    def _strip_think(text: str) -> str:
        """Elimina etiquetas de pensamiento (<think>) de la respuesta del modelo.

        Args:
            text: Texto de la respuesta del modelo.

        Returns:
            Texto limpio sin etiquetas de pensamiento.
        """
        if not text:
            return text
        if '</think>' in text:
            return text.split('</think>')[-1].strip()
        return text

    def _get_fallback_llm(self) -> Optional[GeminiLLM]:
        """Inicializa el cliente Gemini solo si es necesario (lazy initialization).

        Returns:
            Instancia de GeminiLLM si se puede inicializar, o None si
            la clave de API no está disponible o la inicialización falla.
        """
        if self._fallback_llm is not None:
            return self._fallback_llm
        if not GEMINI_API_KEY:
            logger.warning("Gemini solicitado pero GEMINI_API_KEY no está configurada")
            return None
        try:
            self._fallback_llm = GeminiLLM(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL,
                                           max_tokens=self.max_tokens, temperature=self.temperature)
            logger.info("✅ GeminiLLM inicializado (modo directo)" if self._prefer_gemini
                        else "✅ Fallback GeminiLLM inicializado para LangChainVLLMAdapter")
        except Exception as exc:
            logger.error(f"❌ No se pudo inicializar GeminiLLM: {exc}")
            self._fallback_llm = None
        return self._fallback_llm

    @staticmethod
    def _compose_fallback_prompt(prompt: str, system_instruction: Optional[str]) -> str:
        """Compone un prompt para el fallback de Gemini combinando instrucción y entrada.

        Args:
            prompt: Texto del prompt del usuario.
            system_instruction: Instrucción de sistema opcional a anteponer.

        Returns:
            Cadena formateada con la instrucción de sistema (si existe) y
            el prompt del usuario.
        """
        if system_instruction:
            return f"[Instrucción del sistema]\n{system_instruction}\n\n[Entrada del usuario]\n{prompt}"
        return prompt

    def generate(self, prompt: str, system_instruction: str = None) -> str:
        """Genera una respuesta a partir de un prompt, con fallback a Gemini.

        Si USE_GEMINI está activado, usa Gemini directamente. En caso
        contrario, intenta con vLLM vía LangChain y cae a Gemini si falla.

        Args:
            prompt: Texto de entrada para la generación.
            system_instruction: Instrucción de sistema opcional.

        Returns:
            Cadena con la respuesta generada.

        Raises:
            RuntimeError: Si el motor preferido no está disponible y no
                hay fallback configurado.
        """
        msgs = []
        if system_instruction:
            msgs.append(SystemMessage(content=system_instruction))
        else:
            msgs.append(SystemMessage(content=DEFAULT_SYSTEM_ES))
        msgs.append(HumanMessage(content=prompt))

        if self._prefer_gemini:
            fallback = self._get_fallback_llm()
            if not fallback:
                raise RuntimeError("Gemini no disponible para modo USE_GEMINI=1")
            return fallback.generate(self._compose_fallback_prompt(prompt, system_instruction))

        if self.llm is None:
            raise RuntimeError("Cliente vLLM no inicializado y USE_GEMINI=0")

        try:
            resp = self.llm.invoke(msgs)
            return self._strip_think(resp.content)
        except Exception as exc:
            logger.warning(f"⚠️ LangChainVLLMAdapter.generate falló: {exc}. Probando fallback Gemini...")
            if not self._fallback_enabled:
                raise
            fallback = self._get_fallback_llm()
            if fallback is None:
                raise
            return fallback.generate(self._compose_fallback_prompt(prompt, system_instruction))

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Genera una respuesta de chat a partir de un historial de mensajes.

        Convierte los mensajes al formato LangChain y los envía al modelo.
        Si USE_GEMINI está activado o si vLLM falla, recurre a Gemini
        formateando la conversación con roles CLIENTE y ASESOR.

        Args:
            messages: Lista de diccionarios con 'role' ('system', 'user',
                'assistant') y 'content'.

        Returns:
            Cadena con la respuesta generada.

        Raises:
            RuntimeError: Si el motor preferido no está disponible y no
                hay fallback configurado.
        """
        role_map = {
            'system': SystemMessage,
            'user': HumanMessage,
            'assistant': AIMessage,
        }
        msgs = [role_map[m['role']](content=m['content']) for m in messages if m['role'] in role_map]
        if not any(isinstance(m, SystemMessage) for m in msgs):
            msgs.insert(0, SystemMessage(content=DEFAULT_SYSTEM_ES))

        if self._prefer_gemini:
            fallback = self._get_fallback_llm()
            if not fallback:
                raise RuntimeError("Gemini no disponible para modo USE_GEMINI=1")
            prompt_lines = []
            system_instruction = None
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    prompt_lines.append(f"CLIENTE: {content}")
                elif role == "assistant":
                    prompt_lines.append(f"ASESOR: {content}")
            prompt_lines.append("ASESOR:")
            prompt = "\n".join(prompt_lines)
            return fallback.generate(self._compose_fallback_prompt(prompt, system_instruction))

        if self.llm is None:
            raise RuntimeError("Cliente vLLM no inicializado y USE_GEMINI=0")

        try:
            resp = self.llm.invoke(msgs)
            return self._strip_think(resp.content)
        except Exception as exc:
            logger.warning(f"⚠️ LangChainVLLMAdapter.chat falló: {exc}. Probando fallback Gemini...")
            if not self._fallback_enabled:
                raise
            fallback = self._get_fallback_llm()
            if fallback is None:
                raise
            # Convertir mensajes a un prompt simple para Gemini
            prompt_lines = []
            system_instruction = None
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    system_instruction = content
                elif role == "user":
                    prompt_lines.append(f"CLIENTE: {content}")
                elif role == "assistant":
                    prompt_lines.append(f"ASESOR: {content}")
            prompt_lines.append("ASESOR:")
            prompt = "\n".join(prompt_lines)
            return fallback.generate(self._compose_fallback_prompt(prompt, system_instruction))

    # ================================
    # BATCH METHODS (compatibilidad)
    # ================================

    def batch_generate(self, prompts: List[str], system_instruction: str = None) -> List[str]:
        """Genera respuestas en lote de forma síncrona (compatibilidad).

        Delega en batch_generate_async_wrapper para procesamiento paralelo
        mediante ThreadPoolExecutor.

        Args:
            prompts: Lista de cadenas con los prompts a procesar.
            system_instruction: Instrucción de sistema opcional.

        Returns:
            Lista de cadenas con las respuestas generadas.
        """
        return self.batch_generate_async_wrapper(prompts=prompts, system_instruction=system_instruction)

    def batch_generate_async_wrapper(self, prompts: List[str], system_instruction: str = None,
                                     max_workers: Optional[int] = None) -> List[str]:
        """Procesa un listado de prompts en paralelo y devuelve una lista de respuestas.

        Compatible con fiscal_classifier. Implementado mediante ThreadPoolExecutor
        para paralelismo simple sin dependencia de asyncio.

        Args:
            prompts: Lista de cadenas con los prompts a procesar.
            system_instruction: Instrucción de sistema opcional.
            max_workers: Número máximo de hilos. Si es None, usa min(8, len(prompts)).

        Returns:
            Lista de cadenas con las respuestas generadas, en el mismo
            orden que los prompts de entrada.
        """
        if not prompts:
            return []

        def _one(p: str) -> str:
            try:
                return self.generate(p, system_instruction=system_instruction)
            except Exception as e:
                logger.error(f"LangChainVLLMAdapter batch error: {e}")
                return f"Error: {e}"

        if max_workers is None:
            # Conservador para no saturar el backend; ajustable por env
            max_workers = min(8, len(prompts))

        if max_workers <= 1:
            return [_one(p) for p in prompts]

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            return list(ex.map(_one, prompts))
