from django.test import SimpleTestCase

from monitor.vision.clasificador_epp import clasificar_casco, clasificar_mascarilla
from monitor.vision.detector_rostros import Rostro

from .utilidades import (
    AMARILLO_CASCO, AZUL_MASCARILLA, BLANCO_MASCARILLA, PIEL_OSCURA, escena, rostro_sintetico,
)


class MascarillaTests(SimpleTestCase):
    def test_rostro_descubierto_es_sin_mascarilla(self):
        estado = clasificar_mascarilla(escena(), rostro_sintetico())
        self.assertIs(estado.presente, False)
        self.assertGreater(estado.puntaje, 50)

    def test_mascarilla_azul(self):
        estado = clasificar_mascarilla(escena(mascarilla=AZUL_MASCARILLA), rostro_sintetico())
        self.assertIs(estado.presente, True)

    def test_mascarilla_blanca(self):
        estado = clasificar_mascarilla(escena(mascarilla=BLANCO_MASCARILLA), rostro_sintetico())
        self.assertIs(estado.presente, True)

    def test_funciona_con_piel_oscura(self):
        self.assertIs(clasificar_mascarilla(escena(piel=PIEL_OSCURA), rostro_sintetico()).presente, False)
        self.assertIs(
            clasificar_mascarilla(escena(piel=PIEL_OSCURA, mascarilla=AZUL_MASCARILLA), rostro_sintetico()).presente,
            True,
        )

    def test_rostro_lejano_es_indeterminado(self):
        lejano = Rostro((300, 300, 25, 30), 90.0, None)
        self.assertIsNone(clasificar_mascarilla(escena(), lejano).presente)
        self.assertIsNone(clasificar_casco(escena(), lejano).presente)

    def test_sin_puntos_usa_proporciones_del_rostro(self):
        rostro = Rostro((220, 200, 200, 200), 100.0, None)  # como Haar Cascade
        self.assertIs(clasificar_mascarilla(escena(mascarilla=AZUL_MASCARILLA), rostro).presente, True)
        self.assertIs(clasificar_mascarilla(escena(), rostro).presente, False)


class CascoTests(SimpleTestCase):
    def test_casco_amarillo(self):
        estado = clasificar_casco(escena(sobre_cabeza=AMARILLO_CASCO), rostro_sintetico())
        self.assertIs(estado.presente, True)

    def test_cabello_es_sin_casco(self):
        self.assertIs(clasificar_casco(escena(), rostro_sintetico()).presente, False)

    def test_cabeza_cortada_por_el_borde_es_indeterminado(self):
        rostro = rostro_sintetico(y=10)
        self.assertIsNone(clasificar_casco(escena(y=10), rostro).presente)
