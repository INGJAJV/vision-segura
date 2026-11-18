from django.db import models
from django.utils import timezone

CELULAR = "celular"
SIN_MASCARILLA = "sin_mascarilla"
SIN_CASCO = "sin_casco"

TIPOS_INFRACCION = [
    (CELULAR, "Uso de celular"),
    (SIN_MASCARILLA, "Persona sin mascarilla"),
    (SIN_CASCO, "Persona sin casco"),
]


class Regla(models.Model):
    """Norma del laboratorio que el sistema vigila. Se activa/desactiva en vivo."""

    clave = models.CharField(max_length=30, choices=TIPOS_INFRACCION, unique=True)
    descripcion = models.CharField(max_length=200, blank=True)
    activa = models.BooleanField(default=True)

    class Meta:
        ordering = ["clave"]

    def __str__(self):
        return self.get_clave_display()


class Incidencia(models.Model):
    """Infracción confirmada por el sistema, con foto del momento."""

    tipo = models.CharField(max_length=30, choices=TIPOS_INFRACCION, db_index=True)
    fecha_hora = models.DateTimeField(default=timezone.now, db_index=True)
    confianza = models.FloatField(help_text="Confianza de la detección (0-100).")
    personas = models.PositiveIntegerField(default=0, help_text="Personas en cámara.")
    captura = models.FileField(upload_to="incidencias/%Y/%m/%d/", blank=True)
    revisada = models.BooleanField(default=False)

    class Meta:
        ordering = ["-fecha_hora"]

    def __str__(self):
        return f"{self.get_tipo_display()} – {timezone.localtime(self.fecha_hora):%Y-%m-%d %H:%M:%S}"
