"""Une detectores y clasificadores: analiza un frame y dibuja el resultado.

No guarda estado entre frames, por eso sirve tanto para el video en vivo
como para analizar una imagen subida por el usuario.
"""
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import procesamiento as proc
from .clasificador_epp import EstadoEPP, clasificar_casco, clasificar_mascarilla
from .detector_rostros import Rostro

# Colores en BGR
VERDE = (80, 200, 60)
ROJO = (60, 60, 230)
NARANJA = (0, 150, 255)
AZUL = (230, 160, 40)
GRIS = (170, 170, 170)
BLANCO = (255, 255, 255)

COLORES_OBJETO = {"cell phone": ROJO, "person": GRIS}


@dataclass
class PersonaEPP:
    rostro: Rostro
    mascarilla: EstadoEPP
    casco: EstadoEPP | None = None


@dataclass
class Resultado:
    objetos: list = field(default_factory=list)
    personas: list[PersonaEPP] = field(default_factory=list)
    brillo: float = 0.0
    iluminacion_mejorada: bool = False

    @property
    def num_personas(self) -> int:
        cuerpos = sum(1 for o in self.objetos if o.clase == "person")
        return max(cuerpos, len(self.personas))

    def conteo_objetos(self) -> dict[str, int]:
        return dict(Counter(o.etiqueta for o in self.objetos if o.clase != "person"))

    def infracciones(self) -> dict[str, float]:
        """Infracciones visibles en este frame -> confianza máxima (0-100)."""
        resultado = {}
        celulares = [o.confianza for o in self.objetos if o.clase == "cell phone"]
        if celulares:
            resultado["celular"] = max(celulares)
        sin_mascarilla = [p.mascarilla.puntaje for p in self.personas if p.mascarilla.presente is False]
        if sin_mascarilla:
            resultado["sin_mascarilla"] = max(sin_mascarilla)
        sin_casco = [p.casco.puntaje for p in self.personas if p.casco and p.casco.presente is False]
        if sin_casco:
            resultado["sin_casco"] = max(sin_casco)
        return resultado

    def a_dict(self) -> dict:
        return {
            "personas": self.num_personas,
            "objetos": [
                {"etiqueta": o.etiqueta, "confianza": o.confianza} for o in self.objetos
            ],
            "rostros": [
                {
                    "mascarilla": p.mascarilla.presente,
                    "mascarilla_puntaje": p.mascarilla.puntaje,
                    "casco": None if p.casco is None else p.casco.presente,
                }
                for p in self.personas
            ],
            "brillo": round(self.brillo, 1),
            "iluminacion_mejorada": self.iluminacion_mejorada,
        }


class Analizador:
    def __init__(self, detector_rostros, detector_objetos=None, config=None):
        self.detector_rostros = detector_rostros
        self.detector_objetos = detector_objetos
        self.config = {
            "UMBRAL_MASCARILLA": 0.45,
            "UMBRAL_CASCO": 0.40,
            "BRILLO_MINIMO": 70,
            **(config or {}),
        }

    def analizar(self, frame: np.ndarray, evaluar_casco: bool = True) -> Resultado:
        brillo = proc.brillo(frame)
        mejorada = brillo < self.config["BRILLO_MINIMO"]
        imagen = proc.mejorar_iluminacion(frame) if mejorada else frame

        objetos = self.detector_objetos.detectar(imagen) if self.detector_objetos else []
        personas = []
        for rostro in self.detector_rostros.detectar(imagen):
            mascarilla = clasificar_mascarilla(imagen, rostro, self.config["UMBRAL_MASCARILLA"])
            casco = clasificar_casco(imagen, rostro, self.config["UMBRAL_CASCO"]) if evaluar_casco else None
            personas.append(PersonaEPP(rostro, mascarilla, casco))
        return Resultado(objetos, personas, brillo, mejorada)

    # ------------------------------------------------------------- dibujo
    def dibujar(self, frame: np.ndarray, resultado: Resultado) -> np.ndarray:
        for obj in resultado.objetos:
            color = COLORES_OBJETO.get(obj.clase, AZUL)
            grosor = 1 if obj.clase == "person" else 2
            texto = f"{obj.etiqueta} {obj.confianza:.0f}%"
            dibujar_caja(frame, obj.caja, texto, color, grosor)

        for p in resultado.personas:
            lineas = []
            if p.mascarilla.presente is True:
                lineas.append(("Con mascarilla", VERDE))
            elif p.mascarilla.presente is False:
                lineas.append(("Sin mascarilla", ROJO))
            if p.casco is not None and p.casco.presente is not None:
                lineas.append(("Con casco", VERDE) if p.casco.presente else ("Sin casco", ROJO))
            color = ROJO if any(c == ROJO for _, c in lineas) else VERDE
            x, y, w, h = p.rostro.caja
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            for i, (texto, c) in enumerate(lineas):
                dibujar_etiqueta(frame, (x, y + h + 4 + i * 22), texto, c, abajo=True)
        return frame


def dibujar_etiqueta(frame, origen, texto, color, abajo=False):
    texto = proc.texto_ascii(texto)
    x, y = origen
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    if abajo:
        y0, y1 = y, y + th + 8
    else:
        y1 = max(y, th + 8)
        y0 = y1 - th - 8
    cv2.rectangle(frame, (x, y0), (x + tw + 8, y1), color, -1)
    cv2.putText(frame, texto, (x + 4, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANCO, 1, cv2.LINE_AA)


def dibujar_caja(frame, caja, texto, color, grosor=2):
    x, y, w, h = caja
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, grosor)
    if texto:
        dibujar_etiqueta(frame, (x, y), texto, color)
