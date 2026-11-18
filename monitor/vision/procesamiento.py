"""Procesamiento de imágenes: iluminación, recortes y segmentación por color."""
import unicodedata

import cv2
import numpy as np

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

# Rango de piel humana en el espacio YCrCb (independiente del brillo Y).
# Funciona para tonos de piel claros y oscuros porque solo mira la crominancia.
PIEL_MIN = np.array([0, 133, 77], dtype=np.uint8)
PIEL_MAX = np.array([255, 173, 127], dtype=np.uint8)
# Los labios y la piel enrojecida tienen más componente roja (Cr). Se usa este
# rango ampliado en la zona de la boca; las mascarillas (azul, blanca, negra)
# siguen fuera porque su Cr es <= 128.
PIEL_LABIOS_MAX = np.array([255, 195, 127], dtype=np.uint8)


def a_gris(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def brillo(frame: np.ndarray) -> float:
    return float(a_gris(frame).mean())


def mejorar_iluminacion(frame: np.ndarray) -> np.ndarray:
    """Aplica CLAHE al canal de luminancia (LAB) sin alterar los colores."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge((_clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def recortar(imagen: np.ndarray, caja, margen: float = 0.0) -> np.ndarray:
    """Recorta (x, y, w, h) con margen opcional sin salirse de la imagen."""
    x, y, w, h = (int(v) for v in caja)
    dx, dy = int(w * margen), int(h * margen)
    alto, ancho = imagen.shape[:2]
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
    return imagen[y0:y1, x0:x1]


def recortar_coords(imagen: np.ndarray, x0, y0, x1, y1) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(ancho, int(x1)), min(alto, int(y1))
    if x1 <= x0 or y1 <= y0:
        return imagen[0:0, 0:0]
    return imagen[y0:y1, x0:x1]


def mascara_piel(bgr: np.ndarray, incluir_labios: bool = False) -> np.ndarray:
    """Máscara binaria de píxeles con color de piel (con apertura morfológica)."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mascara = cv2.inRange(ycrcb, PIEL_MIN, PIEL_LABIOS_MAX if incluir_labios else PIEL_MAX)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo)


def proporcion_piel(bgr: np.ndarray, incluir_labios: bool = False) -> float:
    if bgr.size == 0:
        return 0.0
    return float(np.count_nonzero(mascara_piel(bgr, incluir_labios))) / (bgr.shape[0] * bgr.shape[1])


def proporcion_color_casco(bgr: np.ndarray) -> float:
    """Fracción de píxeles con colores típicos de casco de seguridad.

    Cascos: colores saturados (amarillo, naranja, rojo, azul, verde) o blanco
    brillante. El cabello es oscuro y la piel se excluye explícitamente.
    """
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    saturado = (s >= 90) & (v >= 90)
    blanco = (s < 40) & (v >= 200)
    no_piel = mascara_piel(bgr) == 0
    candidatos = (saturado | blanco) & no_piel
    return float(np.count_nonzero(candidatos)) / candidatos.size


def redimensionar_max(frame: np.ndarray, ancho_max: int = 960) -> np.ndarray:
    alto, ancho = frame.shape[:2]
    if ancho <= ancho_max:
        return frame
    escala = ancho_max / ancho
    return cv2.resize(frame, (ancho_max, int(alto * escala)), interpolation=cv2.INTER_AREA)


def texto_ascii(texto: str) -> str:
    """cv2.putText solo dibuja ASCII: 'Mascarilla sí' -> 'Mascarilla si'."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def codificar_jpeg(frame: np.ndarray, calidad: int = 80) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    if not ok:
        raise ValueError("No se pudo codificar el frame a JPEG")
    return buffer.tobytes()


def decodificar_imagen(datos: bytes) -> np.ndarray:
    imagen = cv2.imdecode(np.frombuffer(datos, dtype=np.uint8), cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError("El archivo no es una imagen válida")
    return imagen
