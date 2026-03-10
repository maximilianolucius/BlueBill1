#!/usr/bin/env python3
import os

from dotenv import load_dotenv
from openai import OpenAI

QUESTION = "Que dia es hoy?"


def main() -> None:
    load_dotenv()
    base_url = os.getenv("VLLM_BASE_URL")
    model = os.getenv("VLLM_MODEL")
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")

    if not base_url or not model:
        raise SystemExit("Faltan VLLM_BASE_URL o VLLM_MODEL")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=15)

    print(f"Pregunta: {QUESTION}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Eres un asistente util."},
                {"role": "user", "content": QUESTION},
            ],
            max_tokens=32,
            temperature=0.0,
        )
    except Exception as exc:
        raise SystemExit(f"Error consultando vLLM: {exc}")

    content = ""
    if response.choices:
        content = (response.choices[0].message.content or "").strip()

    if not content:
        raise SystemExit("Respuesta vacia recibida de vLLM")

    print(f"Respuesta: {content}")


if __name__ == "__main__":
    main()
