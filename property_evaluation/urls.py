from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('evaluate/', views.evaluate_property, name='evaluate'),
    path('result/<int:pk>/', views.result, name='result'),
]