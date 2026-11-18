
"""Pruebas funcionales: casos de uso clave contra un servidor HTTP real.

StaticLiveServerTestCase levanta el servidor de Django en un puerto libre y
las pruebas lo consumen como lo haría el navegador: peticiones HTTP reales,
cookies, token CSRF, archivos estáticos, envío de formularios y lectura del
flujo de video MJPEG.

Casos de uso validados:
    CU-01  Inicio: abrir el panel en vivo
    CU-02  Navegación entre secciones
    CU-03  Detección en video: el flujo /video/ entrega fotogramas anotados
    CU-04  Visualización en la interfaz: métricas y alertas en /api/estado/
    CU-05  Activar / desactivar reglas desde la interfaz (con CSRF)
    CU-06  Analizar una imagen subida
    CU-07  Consultar y exportar reportes
"""
import http.client
import json
import shutil
import tempfile
import uuid
from http.cookiejar import CookieJar
from unittest import mock
from urllib.request import HTTPCookieProcessor, Request, build_opener

import cv2
import numpy as np
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import override_settings

from monitor import repositorio
from monitor.models import Incidencia, Regla
from monitor.vision import procesamiento as proc
from monitor.vision.analizador import Analizador
from monitor.vision.servicio import ServicioVision

from .test_integracion import CamaraSintetica, DetectorCelular, DetectorRostrosFijo
from .utilidades import AZUL_MASCARILLA, escena

MEDIA_TEMPORAL = tempfile.mkdtemp()
CONFIG = {"FRAMES_CONFIRMACION": 8, "ENFRIAMIENTO_SEGUNDOS": 30}
REGLAS = {"celular": True, "sin_mascarilla": True, "sin_casco": False}


class Navegador:
    """Cliente HTTP mínimo que guarda cookies, como un navegador."""

    def __init__(self, base):
        self.base = base
        self.cookies = CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookies))

    def abrir(self, ruta, datos=None, cabeceras=None):
        peticion = Request(self.base + ruta, data=datos, headers=cabeceras or {})
        try:
            r = self.opener.open(peticion, timeout=10)
        except Exception as exc:  # HTTPError también es una respuesta
            if not hasattr(exc, "code"):
                raise
            r = exc
        return r.status if hasattr(r, "status") else r.code, r.headers, r.read()

    def get(self, ruta):
        return self.abrir(ruta)

    def csrf(self):
        return next(c.value for c in self.cookies if c.name == "csrftoken")

    def post(self, ruta, datos=b"", cabeceras=None, con_csrf=True):
        cabeceras = dict(cabeceras or {})
        if con_csrf:
            cabeceras["X-CSRFToken"] = self.csrf()
        return self.abrir(ruta, datos, cabeceras)

    def subir_archivo(self, ruta, campo, nombre, contenido, tipo):
        """Envía un formulario multipart/form-data, como el botón «Analizar»."""
        limite = uuid.uuid4().hex
        cuerpo = (
            f"--{limite}\r\n"
            f'Content-Disposition: form-data; name="csrfmiddlewaretoken"\r\n\r\n{self.csrf()}\r\n'
            f"--{limite}\r\n"
            f'Content-Disposition: form-data; name="{campo}"; filename="{nombre}"\r\n'
            f"Content-Type: {tipo}\r\n\r\n"
        ).encode() + contenido + f"\r\n--{limite}--\r\n".encode()
        return self.post(
            ruta, cuerpo, {"Content-Type": f"multipart/form-data; boundary={limite}"}, con_csrf=False
        )


@override_settings(MEDIA_ROOT=MEDIA_TEMPORAL)
class CasosDeUsoTests(StaticLiveServerTestCase):
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(MEDIA_TEMPORAL, ignore_errors=True)

    def setUp(self):
        for clave, activa in REGLAS.items():
            Regla.objects.update_or_create(clave=clave, defaults={"activa": activa})

        analizador = Analizador(DetectorRostrosFijo(), DetectorCelular())
        self.servicio = ServicioVision(CamaraSintetica(), analizador, repositorio, CONFIG)
        self.servicio.recargar_reglas()
        for objetivo, valor in (
            ("monitor.views.obtener_servicio", self.servicio),
            ("monitor.views.obtener_analizador", Analizador(DetectorRostrosFijo())),
        ):
            parche = mock.patch(objetivo, return_value=valor)
            parche.start()
            self.addCleanup(parche.stop)

        self.navegador = Navegador(self.live_server_url)

    def transmitir(self, frame, veces):
        """Simula la cámara: procesa N fotogramas y publica el último JPEG."""
        for _ in range(veces):
            anotado = self.servicio.procesar_frame(frame.copy())
        self.servicio._jpeg = proc.codificar_jpeg(anotado)

    def leer_fotograma_del_video(self):
        """Abre /video/ como lo hace la etiqueta <img> y extrae el primer JPEG."""
        conexion = http.client.HTTPConnection(self.server_thread.host, self.server_thread.port, timeout=10)
        try:
            conexion.request("GET", "/video/")
            r = conexion.getresponse()
            tipo = r.getheader("Content-Type")
            datos = b""
            while b"\xff\xd9" not in datos[datos.find(b"\xff\xd8"):] or b"\xff\xd8" not in datos:
                datos += r.read1(65536)
        finally:
            conexion.close()
        inicio = datos.find(b"\xff\xd8")
        fin = datos.find(b"\xff\xd9", inicio) + 2
        return r.status, tipo, cv2.imdecode(np.frombuffer(datos[inicio:fin], np.uint8), cv2.IMREAD_COLOR)

    # ----------------------------------------------------------------- CU-01
    def test_cu01_inicio_muestra_el_panel_en_vivo(self):
        estado, cabeceras, cuerpo = self.navegador.get("/")
        html = cuerpo.decode()
        self.assertEqual(estado, 200)
        self.assertIn("text/html", cabeceras["Content-Type"])
        self.assertIn("<title>En vivo · VisionSegura</title>", html)
        self.assertIn('id="video" src="/video/"', html)
        for clave in REGLAS:
            self.assertIn(f'data-regla="{clave}"', html)
        self.assertIn("csrftoken", [c.name for c in self.navegador.cookies])

        # Los archivos estáticos de la interfaz se sirven correctamente.
        for recurso in ("/static/monitor/estilos.css", "/static/monitor/panel.js"):
            with self.subTest(recurso=recurso):
                self.assertEqual(self.navegador.get(recurso)[0], 200)

    # ----------------------------------------------------------------- CU-02
    def test_cu02_navegacion_entre_secciones(self):
        secciones = {"/": "Cámara del laboratorio", "/analizar/": "Analizar una imagen", "/reportes/": "Incidencias del"}
        for ruta, texto in secciones.items():
            with self.subTest(ruta=ruta):
                estado, _, cuerpo = self.navegador.get(ruta)
                self.assertEqual(estado, 200)
                self.assertIn(texto, cuerpo.decode())

    # ----------------------------------------------------------------- CU-03
    def test_cu03_video_entrega_fotogramas_anotados(self):
        self.transmitir(escena(mascarilla=AZUL_MASCARILLA), 3)
        estado, tipo, fotograma = self.leer_fotograma_del_video()

        self.assertEqual(estado, 200)
        self.assertTrue(tipo.startswith("multipart/x-mixed-replace"))
        self.assertIsNotNone(fotograma, "El flujo debe contener un JPEG válido")
        self.assertEqual(fotograma.shape, (480, 640, 3))
        # Barra superior negra con el resumen (personas, objetos, FPS).
        self.assertLess(fotograma[:20, :].mean(), 80)

    def test_cu03_video_muestra_la_alerta_roja_ante_una_infraccion(self):
        self.transmitir(escena(), CONFIG["FRAMES_CONFIRMACION"])
        _, _, fotograma = self.leer_fotograma_del_video()
        franja = fotograma[-25:, :].reshape(-1, 3).mean(axis=0)  # B, G, R
        self.assertGreater(franja[2], franja[0] + 80, "La franja inferior de alerta debe ser roja")

    # ----------------------------------------------------------------- CU-04
    def test_cu04_interfaz_recibe_metricas_y_alertas(self):
        self.transmitir(escena(), CONFIG["FRAMES_CONFIRMACION"])
        estado, cabeceras, cuerpo = self.navegador.get("/api/estado/")
        datos = json.loads(cuerpo)

        self.assertEqual(estado, 200)
        self.assertEqual(cabeceras["Content-Type"], "application/json")
        self.assertEqual(datos["personas"], 1)
        self.assertEqual(datos["mascarilla"], {"con": 0, "sin": 1})
        self.assertEqual(datos["objetos"], {"Celular": 1})
        self.assertEqual({a["clave"] for a in datos["alertas"]}, {"celular", "sin_mascarilla"})
        self.assertEqual(datos["hoy"]["total"], 2)
        self.assertTrue(all(u["captura"] for u in datos["hoy"]["ultimas"]))

    # ----------------------------------------------------------------- CU-05
    def test_cu05_desactivar_regla_desde_la_interfaz(self):
        self.navegador.get("/")  # obtiene la cookie CSRF, como el navegador
        estado, _, cuerpo = self.navegador.post("/api/reglas/celular/")
        self.assertEqual(estado, 200)
        self.assertEqual(json.loads(cuerpo), {"clave": "celular", "activa": False})
        self.assertFalse(Regla.objects.get(clave="celular").activa)

        self.transmitir(escena(mascarilla=AZUL_MASCARILLA), 20)
        self.assertEqual(Incidencia.objects.filter(tipo="celular").count(), 0)

    def test_cu05_cambiar_regla_sin_token_csrf_es_rechazado(self):
        self.navegador.get("/")
        estado, _, _ = self.navegador.post("/api/reglas/celular/", con_csrf=False)
        self.assertEqual(estado, 403)
        self.assertTrue(Regla.objects.get(clave="celular").activa)

    # ----------------------------------------------------------------- CU-06
    def test_cu06_analizar_imagen_subida(self):
        self.navegador.get("/analizar/")
        foto = proc.codificar_jpeg(escena(), 90)
        estado, _, cuerpo = self.navegador.subir_archivo("/analizar/", "imagen", "foto.jpg", foto, "image/jpeg")
        html = cuerpo.decode()

        self.assertEqual(estado, 200)
        self.assertIn("Persona sin mascarilla", html)
        self.assertIn("data:image/jpeg;base64,", html)

    def test_cu06_archivo_que_no_es_imagen_muestra_error(self):
        self.navegador.get("/analizar/")
        estado, _, cuerpo = self.navegador.subir_archivo(
            "/analizar/", "imagen", "trampa.png", b"esto es texto", "image/png"
        )
        self.assertEqual(estado, 200)
        self.assertIn("no es una imagen válida", cuerpo.decode())

    # ----------------------------------------------------------------- CU-07
    def test_cu07_consultar_y_exportar_reportes(self):
        self.transmitir(escena(), CONFIG["FRAMES_CONFIRMACION"])

        _, _, cuerpo = self.navegador.get("/reportes/?tipo=sin_mascarilla")
        self.assertIn("Persona sin mascarilla", cuerpo.decode())

        estado, cabeceras, cuerpo = self.navegador.get("/reportes/csv/")
        filas = cuerpo.decode("utf-8-sig").strip().splitlines()
        self.assertEqual(estado, 200)
        self.assertIn("attachment", cabeceras["Content-Disposition"])
        self.assertEqual(filas[0].split(",")[:2], ["ID", "Tipo"])
        self.assertEqual(len(filas), 3)  # cabecera + celular + sin mascarilla
