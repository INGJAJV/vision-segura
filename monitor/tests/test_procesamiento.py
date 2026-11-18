import numpy as np
from django.test import SimpleTestCase

from monitor.vision import procesamiento as proc

from .utilidades import AMARILLO_CASCO, AZUL_MASCARILLA, CABELLO, PIEL_CLARA, PIEL_OSCURA


def bloque(color, tam=50):
    return np.full((tam, tam, 3), color, dtype=np.uint8)


class PielTests(SimpleTestCase):
    def test_detecta_piel_clara_y_oscura(self):
        self.assertGreater(proc.proporcion_piel(bloque(PIEL_CLARA)), 0.9)
        self.assertGreater(proc.proporcion_piel(bloque(PIEL_OSCURA)), 0.9)

    def test_mascarilla_azul_no_es_piel(self):
        self.assertLess(proc.proporcion_piel(bloque(AZUL_MASCARILLA)), 0.05)

    def test_labios_cuentan_solo_con_rango_ampliado(self):
        labios = (90, 80, 200)  # BGR rojizo, Cr ~ 187
        self.assertLess(proc.proporcion_piel(bloque(labios)), 0.05)
        self.assertGreater(proc.proporcion_piel(bloque(labios), incluir_labios=True), 0.9)

    def test_imagen_vacia_devuelve_cero(self):
        self.assertEqual(proc.proporcion_piel(np.zeros((0, 0, 3), np.uint8)), 0.0)


class ColorCascoTests(SimpleTestCase):
    def test_amarillo_es_color_de_casco(self):
        self.assertGreater(proc.proporcion_color_casco(bloque(AMARILLO_CASCO)), 0.9)

    def test_cabello_y_piel_no_son_casco(self):
        self.assertLess(proc.proporcion_color_casco(bloque(CABELLO)), 0.05)
        self.assertLess(proc.proporcion_color_casco(bloque(PIEL_CLARA)), 0.05)


class UtilidadesTests(SimpleTestCase):
    def test_recortar_no_sale_del_borde(self):
        img = np.zeros((100, 100, 3), np.uint8)
        self.assertEqual(proc.recortar(img, (90, 90, 50, 50), margen=0.2).shape[:2], (20, 20))

    def test_recortar_coords_invertidas_devuelve_vacio(self):
        img = np.zeros((100, 100, 3), np.uint8)
        self.assertEqual(proc.recortar_coords(img, 50, 50, 10, 10).size, 0)

    def test_mejorar_iluminacion_aumenta_contraste_en_imagen_oscura(self):
        rng = np.random.default_rng(0)
        oscura = rng.integers(10, 40, (120, 160, 3), dtype=np.uint8)
        mejorada = proc.mejorar_iluminacion(oscura)
        self.assertEqual(mejorada.shape, oscura.shape)
        self.assertGreater(proc.a_gris(mejorada).std(), proc.a_gris(oscura).std())

    def test_texto_ascii_quita_tildes(self):
        self.assertEqual(proc.texto_ascii("Cámara sí ñ"), "Camara si n")

    def test_jpeg_ida_y_vuelta(self):
        img = bloque(PIEL_CLARA)
        self.assertEqual(proc.decodificar_imagen(proc.codificar_jpeg(img)).shape, img.shape)

    def test_decodificar_basura_lanza_error(self):
        with self.assertRaises(ValueError):
            proc.decodificar_imagen(b"esto no es una imagen")

    def test_redimensionar_mantiene_proporcion(self):
        grande = np.zeros((1080, 1920, 3), np.uint8)
        self.assertEqual(proc.redimensionar_max(grande, 960).shape[:2], (540, 960))
