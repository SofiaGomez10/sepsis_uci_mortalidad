"""Configuración de pytest: agrega la raíz del proyecto a la ruta de Python
para que las pruebas puedan importar el paquete src."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))