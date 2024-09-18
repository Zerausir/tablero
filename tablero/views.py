from django.http import JsonResponse
from django.views.decorators.http import require_GET


@require_GET
def check_session(request):
    return JsonResponse({'is_authenticated': request.user.is_authenticated})
