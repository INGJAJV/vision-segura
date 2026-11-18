"""Configuración del proyecto VisionAsistencia.

Los valores sensibles se leen de variables de entorno (archivo .env).
Ver .env.example para la lista completa.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-inseguro-cambiar-en-produccion")
DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "monitor",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# --- Base de datos -----------------------------------------------------------
# PostgreSQL es la base de datos del proyecto. DB_ENGINE=sqlite solo existe
# para ejecutar las pruebas automáticas en equipos sin PostgreSQL.
if os.getenv("DB_ENGINE", "postgresql") == "sqlite":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("DB_NAME", "vision_segura"),
            "USER": os.getenv("DB_USER", "postgres"),
            "PASSWORD": os.getenv("DB_PASSWORD", ""),
            "HOST": os.getenv("DB_HOST", "localhost"),
            "PORT": os.getenv("DB_PORT", "5432"),
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "es-ec"
TIME_ZONE = "America/Guayaquil"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Visión por computador ---------------------------------------------------
VISION = {
    # Índice de la cámara (0 = cámara integrada) o ruta/URL de un video.
    "FUENTE_CAMARA": os.getenv("FUENTE_CAMARA", "0"),
    # Carpeta con los modelos pre-entrenados (python manage.py descargar_modelos).
    "DIR_MODELOS": BASE_DIR / "modelos_ia",
    # Confianza mínima (0-1) para aceptar una detección de YOLO.
    "CONFIANZA_OBJETOS": float(os.getenv("CONFIANZA_OBJETOS", "0.40")),
    # Frames consecutivos con la infracción antes de registrar una incidencia.
    "FRAMES_CONFIRMACION": int(os.getenv("FRAMES_CONFIRMACION", "8")),
    # Segundos mínimos entre dos incidencias del mismo tipo.
    "ENFRIAMIENTO_SEGUNDOS": int(os.getenv("ENFRIAMIENTO_SEGUNDOS", "30")),
    # Proporción de piel en boca/nariz (relativa a la frente) bajo la cual
    # se considera que la persona lleva mascarilla.
    "UMBRAL_MASCARILLA": float(os.getenv("UMBRAL_MASCARILLA", "0.45")),
    # Proporción de "color de casco" sobre la cabeza para considerar casco puesto.
    "UMBRAL_CASCO": float(os.getenv("UMBRAL_CASCO", "0.40")),
    # Brillo medio por debajo del cual se aplica mejora de iluminación (CLAHE).
    "BRILLO_MINIMO": float(os.getenv("BRILLO_MINIMO", "70")),
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {"monitor": {"handlers": ["console"], "level": "INFO"}},
}
