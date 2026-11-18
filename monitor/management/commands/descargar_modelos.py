"""python manage.py descargar_modelos

Descarga los modelos pre-entrenados (no se suben al repositorio por su tamaño).
"""
import urllib.request
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

MODELOS = {
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


class Command(BaseCommand):
    help = "Descarga YOLOv4-tiny y YuNet en la carpeta de modelos."

    def add_arguments(self, parser):
        parser.add_argument("--forzar", action="store_true", help="Vuelve a descargar aunque ya existan.")

    def handle(self, *args, forzar=False, **opciones):
        destino = Path(settings.VISION["DIR_MODELOS"])
        destino.mkdir(parents=True, exist_ok=True)
        for nombre, url in MODELOS.items():
            ruta = destino / nombre
            if ruta.exists() and ruta.stat().st_size > 0 and not forzar:
                self.stdout.write(f"✔ {nombre} ya existe")
                continue
            self.stdout.write(f"↓ Descargando {nombre}…")
            temporal = ruta.with_suffix(ruta.suffix + ".part")
            try:
                urllib.request.urlretrieve(url, temporal)
            except OSError as exc:
                temporal.unlink(missing_ok=True)
                raise CommandError(f"No se pudo descargar {nombre}: {exc}") from exc
            temporal.rename(ruta)
            self.stdout.write(self.style.SUCCESS(f"✔ {nombre} ({ruta.stat().st_size / 1e6:.1f} MB)"))
        self.stdout.write(self.style.SUCCESS("Modelos listos."))
