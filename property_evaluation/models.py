# property_evaluation/models.py

from django.db import models


class PropertyEvaluation(models.Model):
    total_area = models.FloatField(verbose_name='Общая площадь')
    living_area = models.FloatField(verbose_name='Жилая площадь', null=True, blank=True)  # ← ДОБАВИТЬ
    kitchen_area = models.FloatField(verbose_name='Площадь кухни')
    rooms_count = models.IntegerField(verbose_name='Количество комнат')
    floor = models.IntegerField(verbose_name='Этаж')
    total_floors = models.IntegerField(verbose_name='Всего этажей')
    year_built = models.IntegerField(verbose_name='Год постройки', null=True, blank=True)

    predicted_price = models.FloatField(verbose_name='Предсказанная цена', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)