# general_report_service/views.py
from django.shortcuts import render
from .dash_app import app  # Importing the Dash app


def dash_view(request):
    return render(request, 'general_report.html')
