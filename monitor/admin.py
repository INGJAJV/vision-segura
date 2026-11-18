from django.contrib import admin

from .models import Incidencia, Regla


@admin.register(Regla)
class ReglaAdmin(admin.ModelAdmin):
    list_display = ("clave", "descripcion", "activa")
    list_editable = ("activa",)


@admin.register(Incidencia)
class IncidenciaAdmin(admin.ModelAdmin):
    list_display = ("tipo", "fecha_hora", "confianza", "personas", "revisada")
    list_filter = ("tipo", "revisada")
    date_hierarchy = "fecha_hora"
