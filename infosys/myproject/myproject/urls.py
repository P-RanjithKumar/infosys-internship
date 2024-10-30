"""
URL configuration for myproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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


from django.urls import path
from django.views.generic import RedirectView
from .views import login_view, registration_view, upload_signature, logout_view

urlpatterns = [
    path('', RedirectView.as_view(url='/login/', permanent=False)),  # Redirect root to login
    path('login/', login_view, name='login'),
    path('register/', registration_view, name='register'),
    path('upload/', upload_signature, name='upload_signature'),
    path('logout/', logout_view, name='logout'),
]


