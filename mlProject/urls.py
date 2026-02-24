from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('property_evaluation.urls')),  # подключаем URLы нашего приложения
]