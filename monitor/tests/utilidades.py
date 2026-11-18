"""Imágenes sintéticas y dobles de prueba compartidos por los tests."""
import numpy as np

from monitor.vision.detector_rostros import Rostro

# Colores BGR verificados dentro / fuera del rango de piel YCrCb.
PIEL_CLARA = (138, 172, 224)
PIEL_OSCURA = (36, 85, 141)
AZUL_MASCARILLA = (200, 120, 40)
BLANCO_MASCARILLA = (240, 240, 240)
AMARILLO_CASCO = (0, 215, 255)
CABELLO = (30, 30, 35)
FONDO = (90, 90, 90)


def rostro_sintetico(y=200):
    """Rostro de 200x200 en (220, y) con 5 puntos de referencia tipo YuNet."""
    puntos = np.array([
        [270, y + 70], [370, y + 70],   # ojos
        [320, y + 120],                 # nariz
        [285, y + 160], [355, y + 160], # comisuras de la boca
    ], dtype=np.float32)
    return Rostro((220, y, 200, 200), 95.0, puntos)


def escena(piel=PIEL_CLARA, mascarilla=None, sobre_cabeza=CABELLO, y=200):
    """Frame 640x480 con un rostro; opcionalmente mascarilla y color sobre la cabeza."""
    frame = np.full((480, 640, 3), FONDO, dtype=np.uint8)
    frame[max(0, y - 110):y, 200:440] = sobre_cabeza
    frame[y:y + 200, 220:420] = piel
    if mascarilla is not None:
        frame[y + 110:y + 195, 230:410] = mascarilla
    return frame
