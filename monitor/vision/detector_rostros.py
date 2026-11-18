"""Detección de rostros: YuNet (DNN, robusto con mascarilla) o Haar como respaldo."""
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .procesamiento import a_gris


@dataclass
class Rostro:
    caja: tuple[int, int, int, int]  # x, y, w, h
    confianza: float  # 0-100
    # 5 puntos (x, y): ojo derecho, ojo izquierdo, nariz, boca der., boca izq.
    puntos: np.ndarray | None = None


class DetectorRostros:
    ARCHIVO_YUNET = "face_detection_yunet_2023mar.onnx"

    def __init__(self, dir_modelos=None, umbral=0.75):
        ruta = Path(dir_modelos or "") / self.ARCHIVO_YUNET
        self._yunet = None
        self._haar = None
        if dir_modelos and ruta.exists():
            self._yunet = cv2.FaceDetectorYN.create(str(ruta), "", (320, 320), umbral, 0.3, 5000)
            self.motor = "YuNet"
        else:
            cascada = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            self._haar = cv2.CascadeClassifier(cascada)
            self.motor = "Haar Cascade"

    def detectar(self, frame: np.ndarray) -> list[Rostro]:
        alto, ancho = frame.shape[:2]
        if self._yunet is not None:
            self._yunet.setInputSize((ancho, alto))
            _, caras = self._yunet.detect(frame)
            if caras is None:
                return []
            rostros = []
            for c in caras:
                x, y, w, h = (int(v) for v in c[:4])
                x, y = max(0, x), max(0, y)
                w, h = min(w, ancho - x), min(h, alto - y)
                if w > 0 and h > 0:
                    rostros.append(Rostro((x, y, w, h), round(float(c[14]) * 100, 1), c[4:14].reshape(5, 2)))
            return rostros

        gris = cv2.equalizeHist(a_gris(frame))
        cajas = self._haar.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
        return [Rostro(tuple(int(v) for v in caja), 100.0) for caja in cajas]
