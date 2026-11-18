"""Acceso a la cámara (o a un archivo de video) con reconexión automática."""
import logging
import time

import cv2

logger = logging.getLogger(__name__)


class Camara:
    MAX_FALLOS = 60  # ~3 s de lecturas fallidas antes de reconectar

    def __init__(self, fuente="0", ancho=640, alto=480, espera_reintento=3.0):
        self.fuente = int(fuente) if str(fuente).isdigit() else str(fuente)
        self.ancho, self.alto = ancho, alto
        self.espera_reintento = espera_reintento
        self._cap = None
        self._fallos = 0
        self._ultimo_intento = -espera_reintento

    @property
    def es_archivo(self) -> bool:
        return isinstance(self.fuente, str)

    @property
    def disponible(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def _abrir(self) -> None:
        self._ultimo_intento = time.monotonic()
        cap = cv2.VideoCapture(self.fuente)
        if not cap.isOpened():
            logger.warning("No se pudo abrir la fuente de video %r", self.fuente)
            cap.release()
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ancho)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.alto)
        self._cap = cap
        self._fallos = 0
        logger.info("Fuente de video %r abierta", self.fuente)

    def leer(self):
        """Devuelve un frame BGR o None si la cámara no está disponible."""
        if not self.disponible:
            if time.monotonic() - self._ultimo_intento < self.espera_reintento:
                return None
            self._abrir()
            if not self.disponible:
                return None

        ok, frame = self._cap.read()
        if ok:
            self._fallos = 0
            # Algunas cámaras ignoran la resolución pedida (p. ej. 1920x1080):
            # se reduce para mantener el procesamiento en tiempo real.
            if frame.shape[1] > self.ancho:
                escala = self.ancho / frame.shape[1]
                frame = cv2.resize(frame, (self.ancho, int(frame.shape[0] * escala)), interpolation=cv2.INTER_AREA)
            return frame
        if self.es_archivo:  # video de prueba: vuelve al inicio y se repite
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            return frame if ok else None
        # Al encenderse, la cámara puede tardar en entregar el primer frame:
        # solo se considera perdida tras muchas lecturas fallidas seguidas.
        self._fallos += 1
        if self._fallos >= self.MAX_FALLOS:
            logger.warning("Se perdió la señal de la cámara; reintentando")
            self.liberar()
        else:
            time.sleep(0.05)
        return None

    def liberar(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
