"""Detección de objetos con YOLOv4-tiny usando el módulo DNN de OpenCV."""
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CLASES_COCO = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# Nombres en español de las clases relevantes en un laboratorio.
TRADUCCIONES = {
    "person": "Persona", "cell phone": "Celular", "laptop": "Laptop", "bottle": "Botella",
    "cup": "Taza", "book": "Libro", "backpack": "Mochila", "handbag": "Bolso",
    "keyboard": "Teclado", "mouse": "Mouse", "tvmonitor": "Monitor", "chair": "Silla",
    "scissors": "Tijeras", "remote": "Control", "clock": "Reloj", "umbrella": "Paraguas",
    "knife": "Cuchillo", "wine glass": "Copa", "diningtable": "Mesa", "tie": "Corbata",
    "suitcase": "Maleta", "sports ball": "Pelota", "pottedplant": "Planta",
    "car": "Auto", "truck": "Camión", "bus": "Bus", "bicycle": "Bicicleta",
    "motorbike": "Moto", "dog": "Perro", "cat": "Gato", "bird": "Pájaro",
    "sofa": "Sofá", "bench": "Banca", "vase": "Florero", "toothbrush": "Cepillo",
}


@dataclass
class Objeto:
    clase: str
    etiqueta: str
    confianza: float  # 0-100
    caja: tuple[int, int, int, int]


class ModeloNoEncontrado(FileNotFoundError):
    pass


class DetectorObjetos:
    ARCHIVO_CFG = "yolov4-tiny.cfg"
    ARCHIVO_PESOS = "yolov4-tiny.weights"

    def __init__(self, dir_modelos, confianza_minima=0.4, umbral_nms=0.4, tamano=416):
        cfg = Path(dir_modelos) / self.ARCHIVO_CFG
        pesos = Path(dir_modelos) / self.ARCHIVO_PESOS
        if not (cfg.exists() and pesos.exists()):
            raise ModeloNoEncontrado(
                f"Faltan {cfg.name} / {pesos.name}. Ejecute: python manage.py descargar_modelos"
            )
        red = cv2.dnn.readNetFromDarknet(str(cfg), str(pesos))
        red.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        red.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._modelo = cv2.dnn_DetectionModel(red)
        self._modelo.setInputParams(size=(tamano, tamano), scale=1 / 255.0, swapRB=True)
        self.confianza_minima = confianza_minima
        self.umbral_nms = umbral_nms
        self._lock = threading.Lock()  # la red DNN no es segura entre hilos

    def detectar(self, frame: np.ndarray) -> list[Objeto]:
        with self._lock:
            ids, confianzas, cajas = self._modelo.detect(
                frame, confThreshold=self.confianza_minima, nmsThreshold=self.umbral_nms
            )
        objetos = []
        for id_clase, conf, caja in zip(np.array(ids).flatten(), np.array(confianzas).flatten(), cajas):
            clase = CLASES_COCO[int(id_clase)]
            objetos.append(Objeto(
                clase=clase,
                etiqueta=TRADUCCIONES.get(clase, clase.capitalize()),
                confianza=round(float(conf) * 100, 1),
                caja=tuple(int(v) for v in caja),
            ))
        return objetos
