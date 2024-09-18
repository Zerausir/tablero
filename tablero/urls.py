"""
URL configuration for tablero project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth.decorators import login_required
from index_service.views import index
from index_service.api.urls import urlpatterns as api_urls
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/login/', LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('accounts/logout/', LogoutView.as_view(next_page='/accounts/login/'), name='logout'),
    path('', login_required(index), name='index'),
    path('api/v1/index_service/', include(api_urls)),
    path('', include('general_report_service.urls')),
    path('', include('gpr_service.urls')),
    path('', include('band_occupation_service.urls')),
    path('django_plotly_dash/', include('django_plotly_dash.urls')),
    path('check_session/', views.check_session, name='check_session'),
]
