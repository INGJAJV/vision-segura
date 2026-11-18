from django.urls import path

from . import views

urlpatterns = [
    path("", views.panel, name="panel"),
    path("analizar/", views.analizar, name="analizar"),
    path("reportes/", views.reportes, name="reportes"),
    path("reportes/csv/", views.exportar_csv, name="exportar_csv"),
    path("incidencias/<int:pk>/revisar/", views.incidencia_revisar, name="incidencia_revisar"),
    path("video/", views.video, name="video"),
    path("api/estado/", views.api_estado, name="api_estado"),
    path("api/reglas/<str:clave>/", views.api_regla, name="api_regla"),
]
