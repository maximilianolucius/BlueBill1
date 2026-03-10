#!/usr/bin/env python3
"""
Cargador de clasificaciones fiscales AEAT por código G.

Proporciona la clase ``ClasificacionesFiscalesAEAT`` que carga y consulta
las clasificaciones fiscales oficiales de la AEAT (códigos G01-G46) desde
un archivo JSON. Incluye un mecanismo de fallback que devuelve una
clasificación genérica cuando el código solicitado no se encuentra en la
base de datos.

Es utilizado por ``fiscal_classifier.py`` para enriquecer las
recomendaciones con información sobre modelos de presentación tributaria.
"""

import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path


class ClasificacionesFiscalesAEAT:
    """Clase para manejar las clasificaciones fiscales de la AEAT.

    Carga las clasificaciones desde un archivo JSON y permite consultar
    la información asociada a cada código G. Si el código no existe en
    la base de datos, devuelve una clasificación genérica de fallback.

    Attributes:
        json_path (str): Ruta al archivo JSON con las clasificaciones.
        clasificaciones (dict): Diccionario con las clasificaciones cargadas,
            indexado por código G (p.ej. ``'G01'``, ``'G46'``).
    """

    def __init__(self, json_path: str = "data/fiscal/clasificaciones_fiscales_aeat.json"):
        """Inicializa el cargador de clasificaciones fiscales.

        Args:
            json_path: Ruta al archivo JSON con las clasificaciones.
                Por defecto busca en ``data/fiscal/clasificaciones_fiscales_aeat.json``.
        """
        self.json_path = json_path
        self.clasificaciones = {}
        self._load_data()

    def _load_data(self) -> None:
        """Carga los datos del archivo JSON.

        Si el archivo no existe o contiene JSON inválido, deja el diccionario
        de clasificaciones vacío y emite un mensaje de advertencia.
        """
        try:
            if not os.path.exists(self.json_path):
                print(f"⚠ Archivo {self.json_path} no encontrado. Usando datos vacíos.")
                return

            with open(self.json_path, 'r', encoding='utf-8') as file:
                self.clasificaciones = json.load(file)
                print(f"✅ Cargadas {len(self.clasificaciones)} clasificaciones fiscales")

        except json.JSONDecodeError as e:
            print(f"❌ Error al decodificar JSON: {e}")
            self.clasificaciones = {}
        except Exception as e:
            print(f"❌ Error al cargar el archivo: {e}")
            self.clasificaciones = {}

    def get_clasificacion(self, codigo_g: str) -> Dict[str, Any]:
        """Obtiene una clasificación específica por código G.

        Args:
            codigo_g: Código de clasificación (p.ej. ``'G01'``, ``'G46'``).
                Se normaliza a mayúsculas automáticamente.

        Returns:
            Diccionario con la información de la clasificación. Si el código
            no existe, devuelve una clasificación genérica de fallback.
        """
        codigo_g = codigo_g.upper()

        if codigo_g in self.clasificaciones:
            return self.clasificaciones[codigo_g]
        else:
            return self._get_fallback_clasificacion(codigo_g)

    def _get_fallback_clasificacion(self, codigo_g: str) -> Dict[str, Any]:
        """Proporciona un fallback para códigos G no definidos en la base de datos.

        Args:
            codigo_g: Código G no encontrado en las clasificaciones.

        Returns:
            Diccionario con una descripción genérica y una nota de advertencia
            indicando que debe consultarse la normativa fiscal vigente.
        """
        return {
            "descripcion": f"Clasificación {codigo_g} - No encuentro modelos aplicables.",
            "nota": f"⚠ ADVERTENCIA: {codigo_g} Consulte la normativa fiscal vigente para determinar las obligaciones exactas."
        }
