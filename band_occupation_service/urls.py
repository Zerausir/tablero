# general_report_service/urls.py
from django.urls import path
from .views import dash_view

urlpatterns = [
    path('sacer-ocupacion-de-bandas/', dash_view, name='band_occupation'),
]
