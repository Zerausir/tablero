# Tablero SACER

Sistema de visualización y análisis de datos para el control del espectro radioeléctrico en Ecuador.

## 📋 Descripción

Tablero SACER es una plataforma web desarrollada en Django que proporciona herramientas interactivas de análisis y
visualización para:

- **SACER RTV**: Visualización de resultados del Sistema Automático de Control del Espectro Radioeléctrico para Radio,
  Televisión y Amplitud Modulada
- **SACER Ocupación de Bandas**: Análisis de ocupación del espectro radioeléctrico
- **GPR (Gestión por Resultados)**: Seguimiento y control de indicadores PACT (Plan Anual de Control de
  Telecomunicaciones)

## 🚀 Características Principales

### SACER RTV

- Visualización de niveles de campo eléctrico por frecuencia
- Mapas de calor interactivos para FM, TV y AM
- Sistema de advertencias y alertas de operación
- Gestión de autorizaciones de suspensión y baja potencia
- Exportación de reportes a Excel

### SACER Ocupación de Bandas

- Análisis de ocupación espectral con umbrales configurables
- Cálculo de porcentajes de ocupación por banda
- Análisis de productos de intermodulación (2do y 3er orden)
- Análisis detallado de frecuencias específicas
- Visualización temporal de datos espectrales

### GPR

- Seguimiento de cumplimiento de indicadores PACT
- Visualización de metas planificadas vs. cumplidas
- Gráficos circulares de progreso por indicador
- Generación automática de Anexo 3.1
- Reportes por año (2024/2025)

## 🛠️ Tecnologías

- **Backend**: Django 5.1.1
- **Frontend**: Dash 2.9.3, Plotly
- **Base de datos**: PostgreSQL (producción), SQLite (desarrollo)
- **Procesamiento de datos**: Pandas, NumPy
- **Servidor ASGI**: Daphne
- **Autenticación**: Sistema de usuarios de Django
- **Estilos**: Bootstrap 4, CSS personalizado

## 📦 Instalación

### Requisitos Previos

- Python 3.8 o superior
- PostgreSQL 12 o superior (para producción)
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**

```bash
git clone https://github.com/Zerausir/tablero.git
cd tablero
```

2. **Crear entorno virtual**

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno**

Crear un archivo `.env` en la raíz del proyecto:

```env
# Django
DJANGO_SECRET_KEY=tu_clave_secreta_aqui
DJANGO_DEBUG=False
DJANGO_SECURE_SSL_REDIRECT=True
ALLOWED_HOSTS=["tudominio.com", "localhost"]

# Base de datos PostgreSQL
DBNAME=nombre_bd
USER=usuario_bd
PASSWORD=contraseña_bd
HOST=localhost
PORT=5432

# Rutas del servidor
SERVER_ROUTE=/ruta/base/servidor
SERVER_ROUTE_BANDAS=/ruta/bandas
SERVER_ROUTE_GPR=/ruta/gpr/2024
SERVER_ROUTE_GPR_2025=/ruta/gpr/2025

# Archivos
FILE_ESTACIONES=estaciones.xlsx
FILE_AUT_SUS=autorizaciones_suspension.xlsx
FILE_AUT_BP=autorizaciones_baja_potencia.xlsx

# Ciudades (formato JSON)
CITIES=["Quito", "Guayaquil", "Cuenca"]

# SSL (para producción)
SECURE_SSL_CERTIFICATE_ROUTE=/ruta/cert.pem
SECURE_SSL_KEY_ROUTE=/ruta/key.pem
```

5. **Aplicar migraciones**

```bash
python manage.py migrate
```

6. **Crear superusuario**

```bash
python manage.py createsuperuser
```

7. **Recolectar archivos estáticos**

```bash
python manage.py collectstatic
```

8. **Ejecutar servidor de desarrollo**

```bash
python manage.py runserver
```

## 🔧 Configuración

### Configuración de Base de Datos

El proyecto utiliza PostgreSQL en producción. La configuración está en `tablero/settings.py`:

```python
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env("DBNAME"),
        "USER": env("USER"),
        "PASSWORD": env("PASSWORD"),
        "HOST": env("HOST"),
        "PORT": env("PORT"),
    }
}
```

### Configuración de Seguridad

Para producción, asegúrate de configurar:

- `SECURE_SSL_REDIRECT=True`
- `SESSION_COOKIE_SECURE=True`
- `CSRF_COOKIE_SECURE=True`
- Certificados SSL válidos

### Configuración de Rutas de Datos

Configurar las rutas según tu estructura de archivos:

```env
# Rutas GPR 2024
RUTA_CCDE_01_ENE=/ruta/2024/CCDE-01/enero
RUTA_CCDE_01_FEB=/ruta/2024/CCDE-01/febrero
# ... más rutas

# Rutas GPR 2025
RUTA_CCDE_01_ENE_2025=/ruta/2025/CCDE-01/enero
# ... más rutas
```

## 📱 Uso

### Acceso al Sistema

1. Navega a `http://localhost:8000` (desarrollo) o tu dominio (producción)
2. Inicia sesión con tus credenciales
3. Selecciona el módulo deseado desde el panel principal

### SACER RTV

1. Selecciona fecha inicial y final
2. Elige la ciudad
3. Marca las opciones de autorizaciones y advertencias
4. Explora los datos en las pestañas FM, TV y AM
5. Selecciona frecuencias específicas para análisis detallado
6. Descarga reportes en Excel

### SACER Ocupación de Bandas

1. Selecciona rango de fechas y ciudad
2. Define el rango de frecuencias
3. Ajusta el umbral de nivel de campo
4. Visualiza el heatmap y scatter plot
5. Realiza análisis de intermodulación configurando rangos de fuente

### GPR

1. Selecciona el año (2024 o 2025)
2. Elige la fecha de corte
3. Visualiza gráficos de cumplimiento global
4. Filtra por indicadores específicos
5. Descarga datos o genera Anexo 3.1

## 🏗️ Estructura del Proyecto

```
tablero/
├── accounts/                 # Gestión de usuarios
├── band_occupation_service/  # Módulo ocupación de bandas
│   ├── dash_app.py          # Aplicación Dash
│   ├── services.py          # Lógica de negocio
│   └── utils.py             # Utilidades
├── general_report_service/   # Módulo SACER RTV
│   ├── dash_app.py
│   ├── services.py
│   └── utils.py
├── gpr_service/             # Módulo GPR
│   ├── dash_app.py
│   ├── services.py
│   └── utils.py
├── index_service/           # Página principal
├── static/                  # Archivos estáticos
│   └── css/
├── tablero/                 # Configuración Django
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── requirements.txt
└── manage.py
```

## 🔐 Seguridad

- Autenticación requerida para todos los módulos
- Sesiones con timeout de 30 minutos
- HTTPS forzado en producción
- Headers de seguridad configurados (HSTS, XSS Protection)
- CSRF Protection habilitado
- Validación de datos de entrada

## 📊 Modelos de Datos

### SACER RTV

- **radio_fm_processed_info**: Datos procesados de FM
- **tv_processed_info**: Datos procesados de TV
- **radio_am_processed_info**: Datos procesados de AM
- **rtv_operation_warnings**: Advertencias de operación

### SACER Ocupación de Bandas

- **band_occupation**: Datos FM
- **band_occupation_am**: Datos AM
- **band_occupation_dn**: Datos denominación

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Copyright (c) 2024-2025 Zerausir. Todos los derechos reservados.

Este software es de uso exclusivo para ARCOTEL (Agencia de Regulación y Control de las Telecomunicaciones del Ecuador)
bajo autorización expresa del autor.

**Términos de uso:**

- El uso, modificación y distribución de este software requiere autorización previa y por escrito del autor.
- ARCOTEL tiene autorización de uso bajo acuerdo específico con el autor.
- Queda prohibida la reproducción, distribución o uso de este software sin autorización del autor.
- El código fuente es propiedad intelectual del autor.

Para solicitar autorización de uso, contactar al autor a través de los canales oficiales.

## 👥 Autores

- **Zerausir** - Creador y desarrollador principal - [GitHub](https://github.com/Zerausir)

## 🙏 Agradecimientos

- ARCOTEL por el soporte institucional
- Equipo de Control del Espectro Radioeléctrico
- Coordinación Zonal 2

## 📞 Soporte

Para soporte técnico o solicitudes de autorización de uso, contactar al autor a través de los canales oficiales de
ARCOTEL.

## 🔄 Changelog

### Versión 2.0 (2025)

- Añadido soporte para año 2025 en GPR
- Mejorado sistema de advertencias y alertas RTV
- Implementado análisis de intermodulación
- Optimizaciones de rendimiento

### Versión 1.0 (2024)

- Lanzamiento inicial
- Módulos SACER RTV y Ocupación de Bandas
- Sistema GPR con indicadores PACT

---

**Nota**: Este README asume configuraciones estándar. Ajusta las rutas y configuraciones según tu entorno específico.