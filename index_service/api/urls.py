from django.urls import path
from . import views

urlpatterns = [
    path('', views.IndexAPIView.as_view(), name='index-api'),
]
