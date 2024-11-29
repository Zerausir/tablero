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
DEBUG = env.bool("DJANGO_DEBUG", default=False)

SECURE_SSL_REDIRECT = env.bool("DJANGO_SECURE_SSL_REDIRECT", default=True)
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_HSTS_SECONDS = env.int("DJANGO_SECURE_HSTS_SECONDS", default=2592000)
SECURE_HSTS_INCLUDE_SUBDOMAINS = env.bool("DJANGO_SECURE_HSTS_INCLUDE_SUBDOMAINS", default=True)
SECURE_HSTS_PRELOAD = env.bool("DJANGO_SECURE_HSTS_PRELOAD", default=True)
SECURE_CONTENT_TYPE_NOSNIFF = env.bool("DJANGO_SECURE_CONTENT_TYPE_NOSNIFF", default=True)
SECURE_BROWSER_XSS_FILTER = env.bool("DJANGO_SECURE_BROWSER_XSS_FILTER", default=True)

SESSION_COOKIE_SECURE = env.bool("DJANGO_SESSION_COOKIE_SECURE", default=True)
CSRF_COOKIE_SECURE = env.bool("DJANGO_CSRF_COOKIE_SECURE", default=True)

SECURE_SSL_CERTIFICATE = env("SECURE_SSL_CERTIFICATE_ROUTE")
SECURE_SSL_KEY = env("SECURE_SSL_KEY_ROUTE")

ALLOWED_HOSTS = json.loads(env("ALLOWED_HOSTS"))
# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "daphne",
    "django.contrib.staticfiles",
    # 3rd-party apps
    "rest_framework",
    "django_plotly_dash.apps.DjangoPlotlyDashConfig",
    "bootstrap4",
    # Local
    "accounts.apps.AccountsConfig",
    "index_service.apps.IndexServiceConfig",
    "general_report_service.apps.GeneralReportServiceConfig",
    "gpr_service.apps.GprServiceConfig",
    "band_occupation_service.apps.BandOccupationServiceConfig",
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
    "tablero.middleware.NoCacheMiddleware",
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
ASGI_APPLICATION = "tablero.asgi.application"

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
        'rest_framework.permissions.IsAuthenticated',
    ],
}

X_FRAME_OPTIONS = 'SAMEORIGIN'

# Increase the maximum upload size to 100 MB
DATA_UPLOAD_MAX_MEMORY_SIZE = None  # 100 MB in bytes

# settings.py

# Define environment variables
DBNAME = env("DBNAME")
USER = env("USER")
PASSWORD = env("PASSWORD")
HOST = env("HOST")
PORT = env("PORT")

OPTIONS = env("OPTIONS", "[]")
CITIES = env("CITIES", "[]")
SERVER_ROUTE = env("SERVER_ROUTE")
SERVER_ROUTE_BANDAS = env("SERVER_ROUTE_BANDAS")
DOWNLOAD_ROUTE = env("DOWNLOAD_ROUTE")
FILE_AUT_SUS = env("FILE_AUT_SUS")
FILE_AUT_BP = env("FILE_AUT_BP")
FILE_ESTACIONES = env("FILE_ESTACIONES")

CITIES1 = config("CITIES1", default={}, cast=lambda v: json.loads(v))
CITIES2 = config("CITIES2", default={}, cast=lambda v: json.loads(v))
MONTH_TRANSLATIONS = config("MONTH_TRANSLATIONS", default={}, cast=lambda v: json.loads(v))

COLUMNS_FM = json.loads(env("COLUMNAS_FM"))
COLUMNS_TV = json.loads(env("COLUMNAS_TV"))
COLUMNS_AM = json.loads(env("COLUMNAS_AM"))
COLUMNS_AUT = json.loads(env("COLUMNAS_AUT"))
COLUMNS_AUTBP = json.loads(env("COLUMNAS_AUTBP"))

COLUMNS_BANDS = json.loads(env("COLUMNAS_BANDAS"))

SERVER_ROUTE_GPR_2023 = env("SERVER_ROUTE_GPR_2023")
SERVER_ROUTE_GPR = env("SERVER_ROUTE_GPR")
FILE_INFORMES_GPR_2023 = env("FILE_INFORMES_GPR_2023")
FILE_INFORMES_GPR = env("FILE_INFORMES_GPR")
FILE_PACT_2024 = env("FILE_PACT_2024")
FILE_INDICADORES_CCDE = env("FILE_INDICADORES_CCDE")
FILE_INDICADORES_CCDR = env("FILE_INDICADORES_CCDR")
FILE_INDICADORES_CCDS = env("FILE_INDICADORES_CCDS")
COLUMNAS_INFORMES_GPR = json.loads(env("COLUMNAS_INFORMES_GPR"))
SERVER_ROUTE_PACT_2024 = env("SERVER_ROUTE_PACT_2024")

INDICADORES_GPR = json.loads(env("INDICADORES_GPR"))
INDICADORES_GPR_CCDE = json.loads(env("INDICADORES_GPR_CCDE"))
INDICADORES_GPR_CCDH = json.loads(env("INDICADORES_GPR_CCDH"))
INDICADORES_GPR_CCDS = json.loads(env("INDICADORES_GPR_CCDS"))
INDICADORES_GPR_CCDR = json.loads(env("INDICADORES_GPR_CCDR"))

RUTA_CCDE_01_ENE = env("RUTA_CCDE_01_ENE")
RUTA_CCDE_01_FEB = env("RUTA_CCDE_01_FEB")
RUTA_CCDE_01_MAR = env("RUTA_CCDE_01_MAR")
RUTA_CCDE_01_ABR = env("RUTA_CCDE_01_ABR")
RUTA_CCDE_01_MAY = env("RUTA_CCDE_01_MAY")
RUTA_CCDE_01_JUN = env("RUTA_CCDE_01_JUN")
RUTA_CCDE_01_JUL = env("RUTA_CCDE_01_JUL")
RUTA_CCDE_01_AGO = env("RUTA_CCDE_01_AGO")
RUTA_CCDE_01_SEP = env("RUTA_CCDE_01_SEP")
RUTA_CCDE_01_OCT = env("RUTA_CCDE_01_OCT")
RUTA_CCDE_01_NOV = env("RUTA_CCDE_01_NOV")
RUTA_CCDE_01_DIC = env("RUTA_CCDE_01_DIC")
RUTA_CCDE_02 = env("RUTA_CCDE_02")
RUTA_CCDE_03 = env("RUTA_CCDE_03")
RUTA_CCDE_04 = env("RUTA_CCDE_04")
RUTA_CCDE_05 = env("RUTA_CCDE_05")
RUTA_CCDE_06 = env("RUTA_CCDE_06")
RUTA_CCDE_07 = env("RUTA_CCDE_07")
RUTA_CCDE_08 = env("RUTA_CCDE_08")
RUTA_CCDE_09 = env("RUTA_CCDE_09")
RUTA_CCDE_10 = env("RUTA_CCDE_10")
RUTA_CCDE_11 = env("RUTA_CCDE_11")

RUTA_CCDH_01 = env("RUTA_CCDH_01")

RUTA_CCDS_01 = env("RUTA_CCDS_01")
RUTA_CCDS_03 = env("RUTA_CCDS_03")
RUTA_CCDS_05 = env("RUTA_CCDS_05")
RUTA_CCDS_08 = env("RUTA_CCDS_08")
RUTA_CCDS_09 = env("RUTA_CCDS_09")
RUTA_CCDS_10 = env("RUTA_CCDS_10")
RUTA_CCDS_11 = env("RUTA_CCDS_11")
RUTA_CCDS_12 = env("RUTA_CCDS_12")
RUTA_CCDS_13 = env("RUTA_CCDS_13")
RUTA_CCDS_16 = env("RUTA_CCDS_16")
RUTA_CCDS_17 = env("RUTA_CCDS_17")
RUTA_CCDS_18 = env("RUTA_CCDS_18")
RUTA_CCDS_23 = env("RUTA_CCDS_23")
RUTA_CCDS_27 = env("RUTA_CCDS_27")
RUTA_CCDS_30 = env("RUTA_CCDS_30")
RUTA_CCDS_31 = env("RUTA_CCDS_31")
RUTA_CCDS_32 = env("RUTA_CCDS_32")

RUTA_CCDR_01 = env("RUTA_CCDR_01")
RUTA_CCDR_04 = env("RUTA_CCDR_04")
RUTA_CCDR_06 = env("RUTA_CCDR_06")

AUTH_USER_MODEL = "accounts.CustomUser"
LOGIN_REDIRECT_URL = '/'
LOGIN_URL = '/accounts/login/'
LOGOUT_REDIRECT_URL = '/'

# Configuración de sesión
SESSION_COOKIE_AGE = 1800  # 30 minutos
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
SESSION_SAVE_EVERY_REQUEST = True
