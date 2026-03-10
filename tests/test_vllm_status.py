import os

import pytest
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@pytest.mark.timeout(15)
def test_vllm_status():
    base_url = os.getenv("VLLM_BASE_URL")
    model = os.getenv("VLLM_MODEL")
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")

    if not base_url or not model:
        pytest.fail("Faltan VLLM_BASE_URL o VLLM_MODEL en .env")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=10)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un asistente útil."},
                {"role": "user", "content": "Confirma que estás operativo con una respuesta breve."},
            ],
            max_tokens=10,
            temperature=0.0,
        )
    except Exception as exc:
        pytest.fail(f"No se pudo conectar con vLLM: {exc}")

    assert response.choices, "vLLM no devolvió choices"
    content = response.choices[0].message.content or ""
    assert content.strip(), "vLLM devolvió una respuesta vacía"
