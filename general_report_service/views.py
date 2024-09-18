# general_report_service/views.py
from django.shortcuts import render
from .dash_app import app  # Importing the Dash app
from tablero.decorators import custom_login_required


@custom_login_required
def dash_view(request):
    return render(request, 'general_report.html')
