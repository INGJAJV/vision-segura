"""Acceso a base de datos usado por el hilo de visión.

Cada función llama a close_old_connections() porque el hilo de video vive
mucho tiempo y no pasa por el ciclo request/response de Django.
"""
import uuid

from django.core.files.base import ContentFile
from django.db import close_old_connections, connection

from .models import Incidencia, Regla


def _conexion_fresca():
    # Dentro de una transacción (p. ej. en los tests) no se debe cerrar la conexión.
    if not connection.in_atomic_block:
        close_old_connections()


def reglas_activas() -> list[str]:
    _conexion_fresca()
    return list(Regla.objects.filter(activa=True).values_list("clave", flat=True))


def registrar_incidencia(tipo: str, confianza: float, personas: int, captura_jpeg: bytes | None) -> Incidencia:
    _conexion_fresca()
    incidencia = Incidencia(tipo=tipo, confianza=round(confianza, 1), personas=personas)
    if captura_jpeg:
        incidencia.captura.save(f"{tipo}_{uuid.uuid4().hex[:10]}.jpg", ContentFile(captura_jpeg), save=False)
    incidencia.save()
    return incidencia
