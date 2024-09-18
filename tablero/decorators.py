from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.urls import reverse


def custom_login_required(view_func):
    @login_required
    def wrapped_view(request, *args, **kwargs):
        if not request.user.is_authenticated:
            return HttpResponseRedirect(reverse('login'))
        return view_func(request, *args, **kwargs)

    return wrapped_view
