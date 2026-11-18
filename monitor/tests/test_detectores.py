"""Pruebas con los modelos reales (se omiten si no se han descargado)."""
import unittest

import numpy as np
from django.conf import settings
from django.test import SimpleTestCase

from monitor.vision.analizador import Analizador
from monitor.vision.detector_objetos import DetectorObjetos, ModeloNoEncontrado
from monitor.vision.detector_rostros import DetectorRostros

DIR = settings.VISION["DIR_MODELOS"]
HAY_YOLO = (DIR / DetectorObjetos.ARCHIVO_PESOS).exists()
HAY_YUNET = (DIR / DetectorRostros.ARCHIVO_YUNET).exists()


class DetectorObjetosTests(SimpleTestCase):
    def test_sin_modelo_lanza_error_claro(self):
        with self.assertRaisesMessage(ModeloNoEncontrado, "descargar_modelos"):
            DetectorObjetos("/carpeta/que/no/existe")

    @unittest.skipUnless(HAY_YOLO, "Modelo YOLO no descargado")
    def test_imagen_vacia_no_tiene_objetos(self):
        detector = DetectorObjetos(DIR)
        self.assertEqual(detector.detectar(np.zeros((480, 640, 3), np.uint8)), [])


class DetectorRostrosTests(SimpleTestCase):
    def test_sin_yunet_usa_haar(self):
        self.assertEqual(DetectorRostros(None).motor, "Haar Cascade")

    @unittest.skipUnless(HAY_YUNET, "Modelo YuNet no descargado")
    def test_yunet_en_imagen_vacia(self):
        detector = DetectorRostros(DIR)
        self.assertEqual(detector.motor, "YuNet")
        self.assertEqual(detector.detectar(np.zeros((480, 640, 3), np.uint8)), [])


class AnalizadorTests(SimpleTestCase):
    def test_imagen_oscura_activa_mejora_de_iluminacion(self):
        analizador = Analizador(DetectorRostros(None))
        resultado = analizador.analizar(np.full((240, 320, 3), 20, np.uint8))
        self.assertTrue(resultado.iluminacion_mejorada)
        self.assertEqual(resultado.infracciones(), {})
