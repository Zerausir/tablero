# index_service/serializers.py
from rest_framework import serializers


class PostDetailSerializer(serializers.Serializer):
    option = serializers.CharField(max_length=100)
