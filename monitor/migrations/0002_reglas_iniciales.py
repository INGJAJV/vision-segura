"""Crea las reglas del laboratorio que el sistema vigila."""
from django.db import migrations

REGLAS = [
    ("celular", "Prohibido usar el celular durante la práctica.", True),
    ("sin_mascarilla", "Uso obligatorio de mascarilla.", True),
    ("sin_casco", "Uso obligatorio de casco (talleres/obra).", False),
]


def crear(apps, schema_editor):
    Regla = apps.get_model("monitor", "Regla")
    for clave, descripcion, activa in REGLAS:
        Regla.objects.get_or_create(clave=clave, defaults={"descripcion": descripcion, "activa": activa})


def borrar(apps, schema_editor):
    apps.get_model("monitor", "Regla").objects.filter(clave__in=[r[0] for r in REGLAS]).delete()


class Migration(migrations.Migration):
    dependencies = [("monitor", "0001_initial")]
    operations = [migrations.RunPython(crear, borrar)]
