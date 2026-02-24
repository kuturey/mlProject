from django.db import models


class PropertyEvaluation(models.Model):
    PROPERTY_TYPES = [
        ('apartment', 'Квартира'),
        ('house', 'Дом'),
        ('commercial', 'Коммерческая'),
    ]

    CONDITION_CHOICES = [
        ('excellent', 'Отличное'),
        ('good', 'Хорошее'),
        ('average', 'Среднее'),
        ('needs_repair', 'Требует ремонта'),
    ]

    # Основные характеристики
    property_type = models.CharField('Тип недвижимости', max_length=20, choices=PROPERTY_TYPES)
    total_area = models.FloatField('Общая площадь (м²)')
    living_area = models.FloatField('Жилая площадь (м²)', null=True, blank=True)
    kitchen_area = models.FloatField('Площадь кухни (м²)', null=True, blank=True)
    rooms_count = models.IntegerField('Количество комнат')
    floor = models.IntegerField('Этаж')
    total_floors = models.IntegerField('Всего этажей в доме')
    year_built = models.IntegerField('Год постройки', null=True, blank=True)

    # Расположение
    city = models.CharField('Город', max_length=100)
    district = models.CharField('Район', max_length=100, blank=True)
    street = models.CharField('Улица', max_length=200, blank=True)

    # Дополнительные характеристики
    condition = models.CharField('Состояние', max_length=20, choices=CONDITION_CHOICES)
    has_balcony = models.BooleanField('Наличие балкона', default=False)
    has_parking = models.BooleanField('Парковка', default=False)
    distance_to_metro = models.FloatField('Расстояние до метро (км)', null=True, blank=True)

    # Результат оценки
    predicted_price = models.DecimalField('Прогнозируемая цена', max_digits=12, decimal_places=2, null=True, blank=True)
    created_at = models.DateTimeField('Дата оценки', auto_now_add=True)

    def __str__(self):
        return f"{self.get_property_type_display()} - {self.total_area}м² - {self.created_at}"

    class Meta:
        verbose_name = 'Оценка недвижимости'
        verbose_name_plural = 'Оценки недвижимости'