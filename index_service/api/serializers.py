from rest_framework import serializers


class OptionSerializer(serializers.Serializer):
    options = serializers.ListField(child=serializers.CharField())
