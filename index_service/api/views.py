import json
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import APIException
from rest_framework import status
from .serializers import OptionSerializer
from django.conf import settings


class IndexAPIView(APIView):
    def get(self, request, format=None):
        try:
            options = json.loads(settings.OPTIONS)
            serializer = OptionSerializer({'options': options}, context={'request': request})
            return Response(serializer.data, status=status.HTTP_200_OK)
        except ValueError as e:
            raise APIException(detail="Error while parsing OPTIONS environment variable.") from e
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
