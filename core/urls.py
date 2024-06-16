"""core URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
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
from django.urls import path,include

from app.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index),
    path('predict/<str:ticker_value>/<str:number_of_days>/', predict),
    path('predict/crypto/<str:ticker_value>/<str:number_of_days>/', predict_cryp),
    path('ticker/', ticker),
    # new added...
    path('ticker/crypto/', crypto_ticker),

# ajax query for dynamic ticker search. 
    path('search/', search),
# api section
    # consider changing path of api from javascript in order to accept the informaation teansmitted from the server.
    path('fetch-data/', fetch_data),
    # path('auth/', include('django.contrib.auth.urls')),
    # path('user/signup/',signup),
    # path('user/logout/',logout_user),
    # path('user/login/',login),
    # path('ticker/search/<str:symbol>', search_ticker),
    path('about/', about),
]  