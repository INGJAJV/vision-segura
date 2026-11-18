# VisionSegura · Monitor de EPP y objetos en el laboratorio

Aplicación web en **Django + OpenCV + PostgreSQL** que analiza el video de una cámara en
tiempo real y detecta:

| Qué detecta | Cómo lo hace (OpenCV) |
|---|---|
| **Objetos** (celular, laptop, botella, mochila, personas… 80 clases) | Red neuronal **YOLOv4-tiny** con `cv2.dnn` + supresión de no-máximos (NMS) |
| **Rostros** con 5 puntos (ojos, nariz, boca) | Detector **YuNet** (`cv2.FaceDetectorYN`), con respaldo **Haar Cascade** |
| **Con / sin mascarilla** | Segmentación de piel en espacio **YCrCb**: compara la piel de la zona nariz-boca con la de la frente de la misma persona |
| **Con / sin casco** | Análisis de color en **HSV** sobre la cabeza (colores saturados o blanco brillante, excluyendo piel y cabello) |
| **Poca luz** | Mejora automática con **CLAHE** sobre el canal L (LAB) |

Sobre las detecciones, el sistema aplica **reglas del laboratorio** (prohibido el celular,
mascarilla obligatoria, casco obligatorio) que se activan o desactivan en vivo. Cuando una
infracción se mantiene varios frames seguidos, se guarda una **incidencia** con su foto en
PostgreSQL.

> Proyecto de la asignatura *Construcción de Software*, UNEMI, Prácticas 3 y 4.
> Tema: **Detección de objetos (mascarilla, casco, celular, etc.)**.

---

## Funcionalidades

- **Panel en vivo** (`/`): video con cajas y etiquetas dibujadas por OpenCV, métricas en
  tiempo real (personas, con/sin mascarilla, FPS, objetos en cámara), alerta visual
  parpadeante, interruptores de reglas y registro de actividad.
- **Analizar imagen** (`/analizar/`): sube una foto y recibe el mismo análisis. Sirve para
  probar casco o mascarilla sin tenerlos a mano.
- **Reportes** (`/reportes/`): incidencias por fecha y tipo, gráfico por hora, miniaturas,
  opción de marcar como revisada y **exportación a CSV** (se abre en Excel).
- **Admin de Django** (`/admin/`): gestión de reglas e incidencias.
- **Estabilidad**: reconexión automática de la cámara, pantalla de "Cámara no disponible",
  un solo hilo de procesamiento compartido por todos los navegadores, y errores aislados
  por frame (un fallo no detiene el video).
- **Anti falsos positivos**: confirmación por *N* frames consecutivos, tiempo de espera
  entre incidencias del mismo tipo, y los rostros lejanos (< 40 px) se marcan como
  indeterminados.

## Arquitectura

```
Cámara ──► Camara (reconexión) ──► ServicioVision (hilo en segundo plano)
                                       │
                                       ├─► Analizador
                                       │     ├─ DetectorObjetos  (YOLOv4-tiny)
                                       │     ├─ DetectorRostros  (YuNet / Haar)
                                       │     └─ clasificador_epp (mascarilla, casco)
                                       ├─► Reglas activas ─► Incidencia (PostgreSQL + foto)
                                       └─► último JPEG ─► /video/ (MJPEG) ─► navegador
Navegador ◄── /api/estado/ (JSON, cada 1 s) ◄── métricas del último frame
```

```
├── config/                  # settings, urls, wsgi
├── monitor/                 # app Django
│   ├── models.py            # Regla, Incidencia
│   ├── views.py             # páginas, video MJPEG, API JSON, CSV
│   ├── repositorio.py       # acceso a BD usado por el hilo de video
│   ├── vision/              # módulo OpenCV (independiente de Django)
│   │   ├── procesamiento.py     # CLAHE, segmentación de piel/color, recortes
│   │   ├── detector_objetos.py  # YOLOv4-tiny (cv2.dnn)
│   │   ├── detector_rostros.py  # YuNet / Haar
│   │   ├── clasificador_epp.py  # mascarilla y casco
│   │   ├── analizador.py        # une detectores y dibuja resultados
│   │   ├── camara.py            # captura con reconexión
│   │   └── servicio.py          # pipeline en tiempo real + reglas
│   ├── management/commands/descargar_modelos.py
│   ├── templates/ static/
│   └── tests/               # 68 pruebas automáticas
├── evidencias/              # log de ejecución de las pruebas
├── modelos_ia/              # modelos descargados (no se suben a GitHub)
├── requirements.txt
└── .env.example
```

## Instalación

Requisitos: **Python 3.10–3.12**, **PostgreSQL 14+** (probado con la 18) y una cámara web.

```bash
# 1. Clonar e instalar dependencias
git clone <URL-del-repositorio>
cd <carpeta>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Crear la base de datos en PostgreSQL
psql -U postgres -c "CREATE DATABASE vision_segura;"

# 3. Configurar variables de entorno
cp .env.example .env               # Windows: copy .env.example .env
#   (editar DB_USER / DB_PASSWORD según su PostgreSQL)

# 4. Migraciones (crea las tablas y las 3 reglas iniciales)
python manage.py migrate

# 5. Descargar los modelos de IA (~25 MB)
python manage.py descargar_modelos

# 6. (Opcional) usuario para /admin
python manage.py createsuperuser

# 7. Ejecutar
python manage.py runserver
```

Abrir **http://127.0.0.1:8000**. En macOS, la primera vez el sistema pedirá permiso de
cámara para la Terminal o VS Code.

**¿No tiene cámara?** Use un video: `FUENTE_CAMARA=ruta/al/video.mp4` en `.env`
(el video se repite en bucle).

## Configuración

Todas las opciones se definen en `.env` (ver `.env.example`):

| Variable | Por defecto | Descripción |
|---|---|---|
| `FUENTE_CAMARA` | `0` | Índice de cámara o ruta de video |
| `CONFIANZA_OBJETOS` | `0.40` | Confianza mínima de YOLO (0–1) |
| `FRAMES_CONFIRMACION` | `8` | Frames seguidos para confirmar una infracción |
| `ENFRIAMIENTO_SEGUNDOS` | `30` | Espera mínima entre incidencias del mismo tipo |
| `UMBRAL_MASCARILLA` | `0.45` | Piel en la boca respecto a la frente para decir "con mascarilla" |
| `UMBRAL_CASCO` | `0.40` | Proporción de color de casco sobre la cabeza |
| `BRILLO_MINIMO` | `70` | Por debajo de este brillo se aplica CLAHE |

## Pruebas

### Automáticas

```bash
python manage.py test monitor -v 2
```

**68 pruebas**, organizadas por nivel:

| Tipo | Archivo | Pruebas | Qué valida |
|---|---|---|---|
| Unitaria | `test_procesamiento.py` | 13 | Segmentación de piel y color de casco, CLAHE, recortes, JPEG |
| Unitaria | `test_clasificador_epp.py` | 9 | Mascarilla y casco con imágenes sintéticas |
| Unitaria | `test_detectores.py` | 5 | YOLOv4-tiny, YuNet y respaldo Haar |
| Unitaria | `test_servicio.py` | 9 | Reglas, confirmación por N frames y enfriamiento (con dobles de prueba) |
| Integración | `test_vistas.py` | 14 | Rutas, API JSON, CSV, formularios y repositorio con PostgreSQL |
| Integración | `test_integracion.py` | 8 | Flujo cámara → servicio → PostgreSQL → API, reportes y CSV |
| Funcional | `test_funcional.py` | 10 | Casos de uso contra un servidor HTTP real (CSRF, MJPEG, subida de archivos) |

El log de la última ejecución está en `evidencias/log_pruebas.txt`.

### Cobertura

```bash
pip install coverage
coverage run manage.py test monitor
coverage report        # resumen en consola (78 % en la última medición)
coverage html          # reporte navegable en htmlcov/index.html
```

La configuración está en `.coveragerc` (mide `monitor` y `config`, sin pruebas ni migraciones).

### Casos de prueba manuales

| # | Caso | Pasos | Resultado esperado |
|---|---|---|---|
| CP-01 | Video en vivo | Abrir `/` | Video con FPS y hora; estado "● En vivo" |
| CP-02 | Detección de celular | Mostrar un celular a la cámara ~1 s | Caja roja "Celular"; alerta; incidencia con foto |
| CP-03 | Sin mascarilla | Mirar a la cámara sin mascarilla | Etiqueta roja "Sin mascarilla"; incidencia |
| CP-04 | Con mascarilla | Ponerse mascarilla | Etiqueta verde "Con mascarilla"; sin incidencia |
| CP-05 | Casco | Activar la regla "casco"; usar casco de color | "Con casco" en verde |
| CP-06 | Desactivar regla | Apagar "Uso de celular" y mostrar celular | Se detecta, pero no genera incidencia |
| CP-07 | Antirrebote | Mantener el celular 1 minuto | Una incidencia cada 30 s, no una por frame |
| CP-08 | Poca luz | Apagar parte de la luz | El panel indica "Luz baja: CLAHE" y sigue detectando |
| CP-09 | Cámara desconectada | Desconectar la cámara USB | Pantalla "Cámara no disponible"; se reconecta sola |
| CP-10 | Analizar imagen | Subir foto con casco/mascarilla | Imagen anotada con el resultado |
| CP-11 | Archivo inválido | Subir un .txt renombrado a .png | Mensaje "no es una imagen válida" |
| CP-12 | Reportes y CSV | Filtrar por fecha y exportar | Tabla, gráfico y CSV con las incidencias |

## Limitaciones conocidas

- La **mascarilla** se decide por color: una mascarilla color piel puede confundirse con
  la piel. Funciona bien con mascarillas azules, blancas, negras y de colores.
- El **casco** se decide por color sobre la cabeza: funciona mejor con cascos de colores
  (amarillo, naranja, rojo, azul) o blancos y con fondo que no tenga esos colores justo
  detrás de la cabeza. Por eso esa regla viene desactivada por defecto.
- YOLOv4-tiny prioriza la velocidad: los objetos muy pequeños o lejanos pueden no detectarse.

## Tecnologías

Python 3.12 · Django 5.2 · OpenCV 4.12 (contrib) · NumPy · PostgreSQL 18 · HTML/CSS/JS sin
frameworks.

## Integrantes

| Nombre |
|---|
| Lara Pincay Eduardo Emanuel |
| Richard Sammir Lopez Rosario |
| Alex David Gualli Cunduri |
| Jesus Adrian Vargas Jama |
| Kebin Geordy Guevara Cobo |
