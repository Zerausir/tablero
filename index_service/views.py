import json
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PostDetailSerializer
from django.conf import settings
from django.contrib.auth.decorators import login_required


class PostDetail(APIView):
    def get(self, request, option, format=None):
        options = json.loads(settings.OPTIONS)

        if option in options:
            serializer = PostDetailSerializer(data={'option': option})
            if serializer.is_valid():
                return Response(serializer.data)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response({'error': 'Option not found'}, status=status.HTTP_404_NOT_FOUND)

@login_required()
def index(request):
    options = json.loads(settings.OPTIONS)
    return render(request, 'index.html', {'options': options})
