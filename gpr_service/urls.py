# general_report_service/urls.py
from django.urls import path
from .views import dash_view

urlpatterns = [
    path('gpr/', dash_view, name='gpr'),
]
