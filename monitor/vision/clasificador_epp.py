"""Clasificación de equipo de protección (EPP) a partir de un rostro detectado.

Se usan reglas de procesamiento de imagen en lugar de una red adicional:

* Mascarilla: se compara cuánta piel hay en la zona nariz-boca contra la
  zona de la frente de la MISMA persona. Si la boca tiene mucha menos piel
  que la frente, está cubierta. Comparar con la propia frente hace que el
  método se adapte al tono de piel y a la iluminación.
* Casco: se analiza la franja sobre la cabeza buscando colores típicos de
  casco (saturados o blanco brillante) que no sean piel ni cabello.
"""
from dataclasses import dataclass

import numpy as np

from .detector_rostros import Rostro
from .procesamiento import proporcion_color_casco, proporcion_piel, recortar_coords

# Rostros más pequeños que esto (personas lejanas) no tienen píxeles
# suficientes para analizar la boca o el casco: se reportan indeterminados.
TAMANO_MINIMO_ROSTRO = 40


@dataclass
class EstadoEPP:
    presente: bool | None  # None = no se puede determinar (rostro cortado, etc.)
    puntaje: float  # 0-100: certeza de la decisión


def _zonas_rostro(rostro: Rostro):
    """Devuelve (zona_boca, zona_frente) como coordenadas x0, y0, x1, y1."""
    x, y, w, h = rostro.caja
    if rostro.puntos is not None:
        ojo_d, ojo_i, nariz, boca_d, boca_i = rostro.puntos
        y_ojos = (ojo_d[1] + ojo_i[1]) / 2
        boca = (
            min(boca_d[0], boca_i[0]) - 0.08 * w, nariz[1] - 0.02 * h,
            max(boca_d[0], boca_i[0]) + 0.08 * w, max(boca_d[1], boca_i[1]) + 0.10 * h,
        )
        frente = (
            min(ojo_d[0], ojo_i[0]), y + 0.05 * h,
            max(ojo_d[0], ojo_i[0]), y_ojos - 0.08 * h,
        )
    else:  # sin puntos (Haar): proporciones típicas de un rostro frontal
        boca = (x + 0.25 * w, y + 0.55 * h, x + 0.75 * w, y + 0.90 * h)
        frente = (x + 0.30 * w, y + 0.08 * h, x + 0.70 * w, y + 0.28 * h)
    return boca, frente


def clasificar_mascarilla(frame: np.ndarray, rostro: Rostro, umbral: float = 0.45) -> EstadoEPP:
    if rostro.caja[2] < TAMANO_MINIMO_ROSTRO:
        return EstadoEPP(None, 0.0)
    boca, frente = _zonas_rostro(rostro)
    zona_boca = recortar_coords(frame, *boca)
    zona_frente = recortar_coords(frame, *frente)
    if zona_boca.size == 0:
        return EstadoEPP(None, 0.0)
    piel_boca = proporcion_piel(zona_boca, incluir_labios=True)
    # Si la frente no es visible (flequillo, gorra) se usa una referencia fija.
    piel_ref = max(proporcion_piel(zona_frente), 0.4) if zona_frente.size else 0.6
    relacion = min(piel_boca / piel_ref, 1.0)
    con_mascarilla = relacion < umbral
    # Puntaje: qué tan lejos está la relación del umbral (más lejos = más seguro).
    if con_mascarilla:
        puntaje = 50 + 50 * (umbral - relacion) / umbral
    else:
        puntaje = 50 + 50 * (relacion - umbral) / (1 - umbral)
    return EstadoEPP(con_mascarilla, round(puntaje, 1))


def clasificar_casco(frame: np.ndarray, rostro: Rostro, umbral: float = 0.40) -> EstadoEPP:
    x, y, w, h = rostro.caja
    if w < TAMANO_MINIMO_ROSTRO:
        return EstadoEPP(None, 0.0)
    zona = recortar_coords(frame, x - 0.1 * w, y - 0.55 * h, x + 1.1 * w, y + 0.05 * h)
    # Si la cabeza está cortada por el borde superior no hay datos suficientes.
    if zona.size == 0 or zona.shape[0] < 0.3 * h:
        return EstadoEPP(None, 0.0)
    proporcion = proporcion_color_casco(zona)
    con_casco = proporcion >= umbral
    if con_casco:
        puntaje = 50 + 50 * (proporcion - umbral) / (1 - umbral)
    else:
        puntaje = 50 + 50 * (umbral - proporcion) / umbral
    return EstadoEPP(con_casco, round(min(puntaje, 100.0), 1))
