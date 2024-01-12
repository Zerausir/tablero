# general_report_service/urls.py
from django.urls import path
from .views import dash_view

urlpatterns = [
    path('reporte-general/', dash_view, name='general_report'),
]
