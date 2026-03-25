from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predecir/', views.predecir, name='predecir'),
    path('resultado/', views.resultado, name='resultado'),
]