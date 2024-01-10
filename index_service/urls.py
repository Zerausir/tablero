from django.urls import path, include
from .views import index

urlpatterns = [
    path('', index, name='index'),
    path('api/v1/', include('index_service.api.urls')),
]
