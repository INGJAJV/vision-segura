"""Lógica de reglas e incidencias del servicio en vivo, con dobles de prueba."""
import numpy as np
from django.test import SimpleTestCase

from monitor.vision.analizador import PersonaEPP, Resultado
from monitor.vision.clasificador_epp import EstadoEPP
from monitor.vision.detector_objetos import Objeto
from monitor.vision.servicio import ServicioVision

from .utilidades import rostro_sintetico

CONFIG = {"FRAMES_CONFIRMACION": 3, "ENFRIAMIENTO_SEGUNDOS": 30}
CELULAR = Objeto("cell phone", "Celular", 88.0, (10, 10, 40, 80))


class AnalizadorFalso:
    detector_objetos = object()

    class detector_rostros:
        motor = "Falso"

    def __init__(self):
        self.resultado = Resultado()
        self.evaluo_casco = None

    def analizar(self, frame, evaluar_casco=True):
        self.evaluo_casco = evaluar_casco
        return self.resultado

    def dibujar(self, frame, resultado):
        return frame


class RepoFalso:
    def __init__(self, reglas=("celular", "sin_mascarilla"), falla=False):
        self.reglas = list(reglas)
        self.incidencias = []
        self.falla = falla

    def reglas_activas(self):
        return self.reglas

    def registrar_incidencia(self, tipo, confianza, personas, captura):
        if self.falla:
            raise RuntimeError("BD caída")
        self.incidencias.append((tipo, confianza, personas))


class CamaraFalsa:
    disponible = True

    def liberar(self):
        pass


class ServicioTests(SimpleTestCase):
    def setUp(self):
        self.analizador = AnalizadorFalso()
        self.repo = RepoFalso()
        self.servicio = ServicioVision(CamaraFalsa(), self.analizador, self.repo, CONFIG)
        self.servicio.recargar_reglas()
        self.frame = np.zeros((480, 640, 3), np.uint8)

    def procesar(self, veces, objetos=None, personas=None):
        self.analizador.resultado = Resultado(objetos or [], personas or [], 120.0, False)
        for _ in range(veces):
            self.servicio.procesar_frame(self.frame.copy())

    def test_no_registra_antes_de_confirmar(self):
        self.procesar(2, [CELULAR])
        self.assertEqual(self.repo.incidencias, [])
        self.assertEqual(self.servicio.alertas, [])

    def test_registra_al_confirmar_n_frames(self):
        self.procesar(3, [CELULAR])
        self.assertEqual(self.repo.incidencias, [("celular", 88.0, 0)])
        self.assertEqual(self.servicio.alertas, ["celular"])

    def test_enfriamiento_evita_duplicados(self):
        self.procesar(20, [CELULAR])
        self.assertEqual(len(self.repo.incidencias), 1)
        self.servicio._ultima_incidencia["celular"] -= 31  # pasan 31 s
        self.procesar(1, [CELULAR])
        self.assertEqual(len(self.repo.incidencias), 2)

    def test_interrupcion_reinicia_la_confirmacion(self):
        self.procesar(2, [CELULAR])
        self.procesar(1, [])
        self.procesar(2, [CELULAR])
        self.assertEqual(self.repo.incidencias, [])

    def test_regla_inactiva_no_genera_incidencia(self):
        self.repo.reglas = ["sin_mascarilla"]
        self.servicio.recargar_reglas()
        self.procesar(5, [CELULAR])
        self.assertEqual(self.repo.incidencias, [])

    def test_persona_sin_mascarilla(self):
        persona = PersonaEPP(rostro_sintetico(), EstadoEPP(False, 80.0))
        self.procesar(3, personas=[persona])
        self.assertEqual(self.repo.incidencias, [("sin_mascarilla", 80.0, 1)])

    def test_casco_solo_se_evalua_si_la_regla_esta_activa(self):
        self.procesar(1)
        self.assertFalse(self.analizador.evaluo_casco)
        self.repo.reglas.append("sin_casco")
        self.servicio.recargar_reglas()
        self.procesar(1)
        self.assertTrue(self.analizador.evaluo_casco)

    def test_error_de_bd_no_detiene_el_video(self):
        self.repo.falla = True
        self.procesar(3, [CELULAR])  # no debe lanzar excepción
        self.assertEqual(self.servicio.eventos[0]["tipo"], "error")

    def test_estado_resume_el_ultimo_frame(self):
        persona = PersonaEPP(rostro_sintetico(), EstadoEPP(True, 90.0))
        botella = Objeto("bottle", "Botella", 70.0, (0, 0, 10, 10))
        self.procesar(1, [CELULAR, botella], [persona])
        estado = self.servicio.estado()
        self.assertEqual(estado["personas"], 1)
        self.assertEqual(estado["objetos"], {"Celular": 1, "Botella": 1})
        self.assertEqual(estado["mascarilla"], {"con": 1, "sin": 0})
