"""
Django settings for tablero project.

Generated by 'django-admin startproject' using Django 5.0.1.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/5.0/ref/settings/
"""

from pathlib import Path
from environs import Env
from decouple import config
import json

env = Env()
env.read_env()

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = env("DJANGO_SECRET_KEY")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = env.bool("DJANGO_DEBUG")

ALLOWED_HOSTS = []

# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # 3rd-party apps
    'rest_framework',
    'django_plotly_dash.apps.DjangoPlotlyDashConfig',
    'bootstrap4',
    # Local
    'index_service.apps.IndexServiceConfig',
    'general_report_service.apps.GeneralReportServiceConfig',
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django_plotly_dash.middleware.BaseMiddleware",
    "django_plotly_dash.middleware.ExternalRedirectionMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "tablero.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "tablero.wsgi.application"

# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Password validation
# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = "/static/"
STATICFILES_DIRS = (str(BASE_DIR.joinpath('static')),)
STATIC_ROOT = str(BASE_DIR.joinpath('staticfiles'))
STATICFILES_FINDERS = [
    "django.contrib.staticfiles.finders.FileSystemFinder",
    "django.contrib.staticfiles.finders.AppDirectoriesFinder",
    "django_plotly_dash.finders.DashAssetFinder",
    "django_plotly_dash.finders.DashComponentFinder",
    "django_plotly_dash.finders.DashAppDirectoryFinder",
]

PLOTLY_COMPONENTS = [

    # Common components (ie within dash itself) are automatically added

    # django-plotly-dash components
    "dpd_components",
    # static support if serving local assets
    "dpd_static_support",

    # Other components, as needed
    "dash_bootstrap_components",
]

PLOTLY_DASH = {

    # Route used for the message pipe websocket connection
    "ws_route": "dpd/ws/channel",

    # Route used for direct http insertion of pipe messages
    "http_route": "dpd/views",

    # Flag controlling existince of http poke endpoint
    "http_poke_enabled": True,

    # Insert data for the demo when migrating
    "insert_demo_migrations": False,

    # Timeout for caching of initial arguments in seconds
    "cache_timeout_initial_arguments": 60,

    # Name of view wrapping function
    "view_decorator": None,

    # Flag to control location of initial argument storage
    "cache_arguments": True,

    # Flag controlling local serving of assets
    "serve_locally": False,
}

# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
}

X_FRAME_OPTIONS = 'SAMEORIGIN'

# Increase the maximum upload size to 10 MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10485760  # 10 MB in bytes

# settings.py

# Define environment variables
OPTIONS = env("OPTIONS", "[]")
CITIES = env("CITIES", "[]")
SERVER_ROUTE = env("SERVER_ROUTE")
DOWNLOAD_ROUTE = env("DOWNLOAD_ROUTE")
FILE_AUT_SUS = env("FILE_AUT_SUS")
FILE_AUT_BP = env("FILE_AUT_BP")
FILE_ESTACIONES = env("FILE_ESTACIONES")

CITIES1 = config("CITIES1", default={}, cast=lambda v: json.loads(v))
MONTH_TRANSLATIONS = config("MONTH_TRANSLATIONS", default={}, cast=lambda v: json.loads(v))

COLUMNS_FM = json.loads(env("COLUMNAS_FM"))
COLUMNS_TV = json.loads(env("COLUMNAS_TV"))
COLUMNS_AM = json.loads(env("COLUMNAS_AM"))
COLUMNS_AUT = json.loads(env("COLUMNAS_AUT"))
COLUMNS_AUTBP = json.loads(env("COLUMNAS_AUTBP"))
