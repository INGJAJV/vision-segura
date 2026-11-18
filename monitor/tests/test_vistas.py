import shutil
import tempfile
from unittest import mock

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from monitor import repositorio
from monitor.models import Incidencia, Regla
from monitor.vision import procesamiento as proc
from monitor.vision.analizador import Analizador
from monitor.vision.detector_rostros import DetectorRostros

MEDIA_TEMPORAL = tempfile.mkdtemp()


class ServicioFalso:
    def __init__(self):
        self.recargas = 0

    def estado(self):
        return {"camara": True, "personas": 0, "alertas": [], "eventos": []}

    def recargar_reglas(self):
        self.recargas += 1


@override_settings(MEDIA_ROOT=MEDIA_TEMPORAL)
class VistasTests(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(MEDIA_TEMPORAL, ignore_errors=True)

    def setUp(self):
        self.servicio = ServicioFalso()
        parche = mock.patch("monitor.views.obtener_servicio", return_value=self.servicio)
        parche.start()
        self.addCleanup(parche.stop)

    def test_migracion_crea_las_reglas(self):
        self.assertEqual(
            set(Regla.objects.values_list("clave", flat=True)),
            {"celular", "sin_mascarilla", "sin_casco"},
        )

    def test_panel_muestra_las_reglas(self):
        r = self.client.get(reverse("panel"))
        self.assertContains(r, "Uso de celular")
        self.assertContains(r, 'data-regla="sin_casco"')

    def test_api_estado_incluye_resumen_del_dia(self):
        Incidencia.objects.create(tipo="celular", confianza=90)
        datos = self.client.get(reverse("api_estado")).json()
        self.assertEqual(datos["hoy"]["total"], 1)
        self.assertEqual(datos["hoy"]["por_tipo"], {"celular": 1})

    def test_activar_y_desactivar_regla(self):
        url = reverse("api_regla", args=["celular"])
        self.assertFalse(self.client.post(url).json()["activa"])
        self.assertFalse(Regla.objects.get(clave="celular").activa)
        self.assertEqual(self.servicio.recargas, 1)
        self.assertTrue(self.client.post(url).json()["activa"])

    def test_regla_solo_acepta_post(self):
        self.assertEqual(self.client.get(reverse("api_regla", args=["celular"])).status_code, 405)

    def test_regla_inexistente_404(self):
        self.assertEqual(self.client.post(reverse("api_regla", args=["nada"])).status_code, 404)

    def test_reportes_y_filtro_por_tipo(self):
        Incidencia.objects.create(tipo="celular", confianza=90)
        Incidencia.objects.create(tipo="sin_mascarilla", confianza=70)
        r = self.client.get(reverse("reportes"), {"tipo": "celular"})
        self.assertEqual(r.context["total"], 1)

    def test_reportes_con_fecha_invalida_usa_hoy(self):
        self.assertEqual(self.client.get(reverse("reportes"), {"fecha": "xx"}).status_code, 200)

    def test_exportar_csv(self):
        Incidencia.objects.create(tipo="celular", confianza=90, personas=2)
        r = self.client.get(reverse("exportar_csv"))
        contenido = r.content.decode("utf-8-sig")
        self.assertIn("attachment", r["Content-Disposition"])
        self.assertIn("Uso de celular", contenido)
        self.assertEqual(len(contenido.strip().splitlines()), 2)

    def test_marcar_revisada(self):
        i = Incidencia.objects.create(tipo="celular", confianza=90)
        self.client.post(reverse("incidencia_revisar", args=[i.pk]))
        i.refresh_from_db()
        self.assertTrue(i.revisada)


@override_settings(MEDIA_ROOT=MEDIA_TEMPORAL)
class AnalizarImagenTests(TestCase):
    def setUp(self):
        analizador = Analizador(DetectorRostros(None))
        parche = mock.patch("monitor.views.obtener_analizador", return_value=analizador)
        parche.start()
        self.addCleanup(parche.stop)

    def test_archivo_que_no_es_imagen(self):
        archivo = SimpleUploadedFile("x.png", b"no soy imagen", content_type="image/png")
        r = self.client.post(reverse("analizar"), {"imagen": archivo})
        self.assertContains(r, "no es una imagen válida")

    def test_imagen_valida_devuelve_resultado(self):
        png = proc.codificar_jpeg(np.full((240, 320, 3), 128, np.uint8))
        archivo = SimpleUploadedFile("foto.jpg", png, content_type="image/jpeg")
        r = self.client.post(reverse("analizar"), {"imagen": archivo})
        self.assertEqual(r.status_code, 200)
        self.assertIn("imagen_b64", r.context)
        self.assertContains(r, "Ninguna infracción detectada")


@override_settings(MEDIA_ROOT=MEDIA_TEMPORAL)
class RepositorioTests(TestCase):
    def test_registrar_incidencia_guarda_la_captura(self):
        jpeg = proc.codificar_jpeg(np.zeros((10, 10, 3), np.uint8))
        i = repositorio.registrar_incidencia("celular", 87.456, 3, jpeg)
        self.assertEqual(i.confianza, 87.5)
        self.assertTrue(i.captura.name.startswith("incidencias/"))

    def test_reglas_activas(self):
        self.assertEqual(set(repositorio.reglas_activas()), {"celular", "sin_mascarilla"})
