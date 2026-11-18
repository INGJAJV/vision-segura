"""Pipeline de visión en tiempo real con reglas e incidencias.

Un único hilo en segundo plano lee la cámara, analiza el frame, dibuja los
resultados y guarda el último JPEG. Todas las pestañas del navegador reciben
ese mismo JPEG: el procesamiento ocurre una sola vez.

Una incidencia se registra cuando una infracción se ve en N frames
consecutivos (evita falsos positivos de un frame suelto) y han pasado al
menos `ENFRIAMIENTO_SEGUNDOS` desde la última del mismo tipo.
"""
import logging
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from . import procesamiento as proc
from .analizador import BLANCO, ROJO, Resultado

logger = logging.getLogger(__name__)

NOMBRES_REGLA = {
    "celular": "Uso de celular",
    "sin_mascarilla": "Persona sin mascarilla",
    "sin_casco": "Persona sin casco",
}


class ServicioVision:
    def __init__(self, camara, analizador, repositorio, config):
        self.camara = camara
        self.analizador = analizador
        self.repo = repositorio
        self.config = config

        self.reglas_activas: set[str] = set()
        self._consecutivos: dict[str, int] = {}
        self._ultima_incidencia: dict[str, float] = {}
        self.alertas: list[str] = []
        self.eventos: deque[dict] = deque(maxlen=15)
        self.ultimo_resultado = Resultado()
        self.fps = 0.0

        self._jpeg: bytes | None = None
        self._condicion = threading.Condition()
        self._hilo: threading.Thread | None = None
        self._activo = False
        self._reglas_leidas = -1e9

    # ------------------------------------------------------------------ ciclo
    def iniciar(self):
        if self._hilo and self._hilo.is_alive():
            return
        self.recargar_reglas()
        if self.analizador.detector_objetos is None:
            self.evento("Modelo YOLO no encontrado: ejecute 'manage.py descargar_modelos'", "error")
        self.evento(f"Sistema iniciado (rostros: {self.analizador.detector_rostros.motor})")
        self._activo = True
        self._hilo = threading.Thread(target=self._bucle, name="vision", daemon=True)
        self._hilo.start()

    def detener(self):
        self._activo = False
        if self._hilo:
            self._hilo.join(timeout=2)
        self.camara.liberar()

    def _bucle(self):
        anterior = time.monotonic()
        while self._activo:
            frame = self.camara.leer()
            if frame is None:
                frame = self._frame_sin_camara()
                time.sleep(0.2)
            else:
                try:
                    frame = self.procesar_frame(frame)
                except Exception:  # un frame con error no debe tumbar el video
                    logger.exception("Error procesando frame")
            ahora = time.monotonic()
            self.fps = round(0.9 * self.fps + 0.1 / max(ahora - anterior, 1e-6), 1)
            anterior = ahora
            with self._condicion:
                self._jpeg = proc.codificar_jpeg(frame)
                self._condicion.notify_all()

    def esperar_jpeg(self, timeout=1.0) -> bytes | None:
        with self._condicion:
            self._condicion.wait(timeout=timeout)
            return self._jpeg

    # -------------------------------------------------------- procesamiento
    def procesar_frame(self, frame: np.ndarray) -> np.ndarray:
        if time.monotonic() - self._reglas_leidas > 15:
            self.recargar_reglas()
        resultado = self.analizador.analizar(frame, evaluar_casco="sin_casco" in self.reglas_activas)
        self.analizador.dibujar(frame, resultado)
        self.ultimo_resultado = resultado
        self._evaluar_reglas(resultado, frame)
        self._dibujar_hud(frame, resultado)
        return frame

    def _evaluar_reglas(self, resultado: Resultado, frame):
        infracciones = resultado.infracciones()
        ahora = time.monotonic()
        alertas = []
        for clave in self.reglas_activas:
            if clave not in infracciones:
                self._consecutivos[clave] = 0
                continue
            self._consecutivos[clave] = self._consecutivos.get(clave, 0) + 1
            if self._consecutivos[clave] < self.config["FRAMES_CONFIRMACION"]:
                continue
            alertas.append(clave)
            ultima = self._ultima_incidencia.get(clave)
            if ultima is not None and ahora - ultima < self.config["ENFRIAMIENTO_SEGUNDOS"]:
                continue
            self._ultima_incidencia[clave] = ahora
            self._registrar(clave, infracciones[clave], resultado.num_personas, frame)
        self.alertas = sorted(alertas)

    def _registrar(self, clave, confianza, personas, frame):
        try:
            self.repo.registrar_incidencia(clave, confianza, personas, proc.codificar_jpeg(frame, 85))
            self.evento(f"Incidencia: {NOMBRES_REGLA.get(clave, clave)} ({confianza:.0f}%)", "alerta")
        except Exception:
            logger.exception("No se pudo registrar la incidencia %s", clave)
            self.evento("Error al guardar una incidencia", "error")

    # ----------------------------------------------------- acciones públicas
    def recargar_reglas(self):
        self._reglas_leidas = time.monotonic()
        try:
            self.reglas_activas = set(self.repo.reglas_activas())
        except Exception:
            logger.exception("No se pudieron leer las reglas")

    def evento(self, texto, tipo="info"):
        self.eventos.appendleft({"hora": datetime.now().strftime("%H:%M:%S"), "texto": texto, "tipo": tipo})

    def estado(self) -> dict:
        r = self.ultimo_resultado
        mascarillas = [p.mascarilla.presente for p in r.personas]
        cascos = [p.casco.presente for p in r.personas if p.casco is not None]
        return {
            "camara": self.camara.disponible,
            "yolo": self.analizador.detector_objetos is not None,
            "motor_rostros": self.analizador.detector_rostros.motor,
            "fps": self.fps,
            "personas": r.num_personas,
            "objetos": r.conteo_objetos(),
            "mascarilla": {"con": mascarillas.count(True), "sin": mascarillas.count(False)},
            "casco": {"con": cascos.count(True), "sin": cascos.count(False)},
            "brillo": round(r.brillo, 1),
            "iluminacion_mejorada": r.iluminacion_mejorada,
            "reglas_activas": sorted(self.reglas_activas),
            "alertas": [{"clave": a, "nombre": NOMBRES_REGLA.get(a, a)} for a in self.alertas],
            "eventos": list(self.eventos),
        }

    # --------------------------------------------------------------- dibujo
    def _dibujar_hud(self, frame, resultado: Resultado):
        ancho = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (ancho, 28), (0, 0, 0), -1)
        texto = f"Personas: {resultado.num_personas} | Objetos: {sum(resultado.conteo_objetos().values())} | {self.fps:.0f} FPS"
        if resultado.iluminacion_mejorada:
            texto += " | Luz baja: CLAHE"
        cv2.putText(frame, texto, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANCO, 1, cv2.LINE_AA)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (ancho - 80, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANCO, 1, cv2.LINE_AA)
        if self.alertas:
            alto = frame.shape[0]
            mensaje = "ALERTA: " + ", ".join(NOMBRES_REGLA.get(a, a) for a in self.alertas)
            cv2.rectangle(frame, (0, alto - 32), (ancho, alto), ROJO, -1)
            cv2.putText(frame, proc.texto_ascii(mensaje), (10, alto - 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLANCO, 2, cv2.LINE_AA)

    @staticmethod
    def _frame_sin_camara():
        frame = np.full((480, 640, 3), 35, dtype=np.uint8)
        cv2.putText(frame, "Camara no disponible", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLANCO, 2, cv2.LINE_AA)
        cv2.putText(frame, "Reintentando conexion...", (185, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        return frame


# ------------------------------------------------------------- singletons
_servicio: ServicioVision | None = None
_analizador = None
_lock = threading.Lock()


def obtener_analizador():
    """Analizador compartido (los modelos se cargan una sola vez)."""
    global _analizador
    with _lock:
        if _analizador is None:
            from django.conf import settings

            from .analizador import Analizador
            from .detector_objetos import DetectorObjetos, ModeloNoEncontrado
            from .detector_rostros import DetectorRostros

            cfg = settings.VISION
            try:
                objetos = DetectorObjetos(cfg["DIR_MODELOS"], cfg["CONFIANZA_OBJETOS"])
            except ModeloNoEncontrado as exc:
                logger.warning("%s", exc)
                objetos = None
            _analizador = Analizador(DetectorRostros(cfg["DIR_MODELOS"]), objetos, cfg)
        return _analizador


def obtener_servicio() -> ServicioVision:
    """Crea (una sola vez) e inicia el servicio de video en vivo."""
    global _servicio
    analizador = obtener_analizador()
    with _lock:
        if _servicio is None:
            from django.conf import settings

            from .. import repositorio
            from .camara import Camara

            _servicio = ServicioVision(
                Camara(settings.VISION["FUENTE_CAMARA"]), analizador, repositorio, settings.VISION
            )
            _servicio.iniciar()
        return _servicio
