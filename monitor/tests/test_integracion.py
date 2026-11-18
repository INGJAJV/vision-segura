"""Pruebas de integración: flujo completo entre los módulos del sistema.

A diferencia de las pruebas unitarias, aquí NO se reemplaza la base de datos:
se conectan el servicio de visión real, el analizador real (clasificador de
EPP), el repositorio, el ORM de Django, PostgreSQL y las vistas web.

Solo se sustituyen la cámara (por fotogramas sintéticos) y el detector de
rostros (por uno que siempre devuelve el mismo rostro), para que las pruebas
sean repetibles sin depender del hardware ni de la iluminación.

Flujo validado:
    cámara -> ServicioVision -> Analizador -> reglas -> repositorio
           -> PostgreSQL -> /api/estado/ , /reportes/ , /reportes/csv/
"""
import shutil
import tempfile
from unittest import mock

from django.test import TestCase, override_settings
from django.urls import reverse

from monitor import repositorio
from monitor.models import Incidencia, Regla
from monitor.vision.analizador import Analizador
from monitor.vision.detector_objetos import Objeto
from monitor.vision.servicio import ServicioVision

from .utilidades import AZUL_MASCARILLA, escena, rostro_sintetico

MEDIA_TEMPORAL = tempfile.mkdtemp()
CONFIG = {"FRAMES_CONFIRMACION": 8, "ENFRIAMIENTO_SEGUNDOS": 30}


class DetectorRostrosFijo:
    """Sustituye a YuNet: siempre encuentra el rostro de la escena sintética."""

    motor = "Sintético"

    def detectar(self, imagen):
        return [rostro_sintetico()]


class DetectorCelular:
    """Sustituye a YOLO: informa un celular con 88 % de confianza."""

    def detectar(self, imagen):
        return [Objeto("cell phone", "Celular", 88.0, (20, 20, 60, 120))]


class CamaraSintetica:
    disponible = True

    def liberar(self):
        pass


@override_settings(MEDIA_ROOT=MEDIA_TEMPORAL)
class IntegracionTests(TestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(MEDIA_TEMPORAL, ignore_errors=True)

    def crear_servicio(self, detector_objetos=None):
        """Servicio de visión real conectado al repositorio real (PostgreSQL)."""
        analizador = Analizador(DetectorRostrosFijo(), detector_objetos)
        servicio = ServicioVision(CamaraSintetica(), analizador, repositorio, CONFIG)
        servicio.recargar_reglas()
        parche = mock.patch("monitor.views.obtener_servicio", return_value=servicio)
        parche.start()
        self.addCleanup(parche.stop)
        return servicio

    @staticmethod
    def procesar(servicio, frame, veces):
        for _ in range(veces):
            servicio.procesar_frame(frame.copy())

    # ----------------------------------------------------------------- rutas
    def test_rutas_principales_responden_y_comparten_navegacion(self):
        self.crear_servicio()
        for nombre in ("panel", "analizar", "reportes"):
            with self.subTest(ruta=nombre):
                r = self.client.get(reverse(nombre))
                self.assertEqual(r.status_code, 200)
                self.assertContains(r, "VisionSegura")
                for enlace in ("panel", "analizar", "reportes"):
                    self.assertContains(r, f'href="{reverse(enlace)}"')

    # --------------------------------------------- visión -> base de datos
    def test_persona_sin_mascarilla_se_guarda_en_la_bd_con_foto(self):
        servicio = self.crear_servicio()
        self.procesar(servicio, escena(), CONFIG["FRAMES_CONFIRMACION"])

        incidencia = Incidencia.objects.get()
        self.assertEqual(incidencia.tipo, "sin_mascarilla")
        self.assertEqual(incidencia.personas, 1)
        self.assertTrue(incidencia.captura.name.startswith("incidencias/"))
        self.assertGreater(incidencia.captura.size, 0)

    def test_persona_con_mascarilla_no_genera_incidencia(self):
        servicio = self.crear_servicio()
        self.procesar(servicio, escena(mascarilla=AZUL_MASCARILLA), 20)
        self.assertEqual(Incidencia.objects.count(), 0)
        self.assertEqual(servicio.estado()["mascarilla"], {"con": 1, "sin": 0})

    def test_enfriamiento_con_bd_real_evita_duplicados(self):
        servicio = self.crear_servicio()
        self.procesar(servicio, escena(), 40)
        self.assertEqual(Incidencia.objects.filter(tipo="sin_mascarilla").count(), 1)

    # ------------------------------------------------ BD -> interfaz web
    def test_incidencia_detectada_aparece_en_api_reportes_y_csv(self):
        servicio = self.crear_servicio(DetectorCelular())
        self.procesar(servicio, escena(mascarilla=AZUL_MASCARILLA), CONFIG["FRAMES_CONFIRMACION"])

        estado = self.client.get(reverse("api_estado")).json()
        self.assertEqual(estado["hoy"]["por_tipo"], {"celular": 1})
        self.assertEqual(estado["objetos"], {"Celular": 1})
        self.assertEqual(estado["alertas"][0]["clave"], "celular")

        reportes = self.client.get(reverse("reportes"))
        self.assertEqual(reportes.context["total"], 1)
        self.assertContains(reportes, "Uso de celular")

        csv = self.client.get(reverse("exportar_csv")).content.decode("utf-8-sig")
        self.assertIn("Uso de celular", csv)
        self.assertIn("88.0", csv)

    # --------------------------------------- interfaz web -> visión
    def test_desactivar_regla_desde_la_web_detiene_las_incidencias(self):
        servicio = self.crear_servicio(DetectorCelular())

        r = self.client.post(reverse("api_regla", args=["celular"]))
        self.assertFalse(r.json()["activa"])
        self.assertNotIn("celular", servicio.reglas_activas)

        self.procesar(servicio, escena(mascarilla=AZUL_MASCARILLA), 20)
        self.assertEqual(Incidencia.objects.count(), 0)
        # El objeto se sigue detectando aunque la regla esté apagada.
        self.assertEqual(servicio.estado()["objetos"], {"Celular": 1})

    def test_marcar_revisada_se_refleja_en_el_csv(self):
        servicio = self.crear_servicio()
        self.procesar(servicio, escena(), CONFIG["FRAMES_CONFIRMACION"])
        incidencia = Incidencia.objects.get()

        self.client.post(reverse("incidencia_revisar", args=[incidencia.pk]))
        csv = self.client.get(reverse("exportar_csv")).content.decode("utf-8-sig")
        fila = csv.strip().splitlines()[1]
        self.assertIn("Persona sin mascarilla", fila)
        self.assertIn("Sí", fila)

    def test_reglas_iniciales_controlan_el_servicio(self):
        servicio = self.crear_servicio()
        self.assertEqual(servicio.reglas_activas, {"celular", "sin_mascarilla"})
        Regla.objects.filter(clave="sin_casco").update(activa=True)
        servicio.recargar_reglas()
        self.assertIn("sin_casco", servicio.reglas_activas)
