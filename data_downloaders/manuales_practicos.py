# Manuales_practicos.py

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote
import requests

# SmartDoc API configuration
BASE_URL = "http://172.24.250.17:8001"
BASE_URL = "http://212.69.86.224:8001"


def sanitize_filename(name):
    # Quita caracteres no válidos para nombres de archivo
    return "".join(c for c in name if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()


def download_pdfs(url, download_folder="/tmp/"):
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Buscar todos los enlaces que terminan en .pdf (ignorando mayúsculas/minúsculas)
    pdf_links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.lower().endswith('.pdf'):
            full_url = urljoin(url, href)
            pdf_links.append((link.get_text(strip=True), full_url))

    print(f"Encontrados {len(pdf_links)} archivos PDF.")

    for text, link in pdf_links:
        filename = sanitize_filename(text) or "manual"
        filename = link.split("/")[-1]
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'
        filepath = os.path.join(download_folder, filename)

        # Evitar sobreescribir archivos con el mismo nombre
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(filepath):
            filepath = os.path.join(download_folder, f"{base}_{counter}{ext}")
            counter += 1

        print(f"Descargando {link} a {filepath} ...")
        pdf_response = requests.get(link)
        pdf_response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(pdf_response.content)

        # Upload document
        files = {'file': open(filepath, 'rb')}
        response = requests.post(f'{BASE_URL}/smartdoc/upload', files=files)
        print(response.json())
        doc_id = response.json()['doc_id']
        print(f"Document uploaded. ID: {doc_id}")
        files['file'].close()

    print("Descarga completada.")


if __name__ == "__main__":
    url = "https://sede.agenciatributaria.gob.es/Sede/manuales-practicos.html"
    download_pdfs(url)
