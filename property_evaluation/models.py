from django.db import models


class PropertyEvaluation(models.Model):
    # Основные параметры
    total_meters = models.FloatField(verbose_name='Общая площадь (м²)')
    kitchen_meters = models.FloatField(verbose_name='Площадь кухни (м²)')
    rooms_count = models.IntegerField(verbose_name='Количество комнат')
    floor = models.IntegerField(verbose_name='Этаж')
    floors_count = models.IntegerField(verbose_name='Всего этажей')
    house_age = models.IntegerField(verbose_name='Возраст дома (лет)', null=True, blank=True)
    year_built = models.IntegerField(verbose_name='Год постройки', null=True, blank=True)

    # Категориальные
    author_type = models.CharField(max_length=50, default='real_estate_agent', verbose_name='Тип автора')
    district = models.CharField(max_length=100, default='Центральный', verbose_name='Район')
    street = models.CharField(max_length=200, default='', verbose_name='Улица')
    house_number = models.CharField(max_length=20, default='', verbose_name='Номер дома')
    underground = models.CharField(max_length=100, default='', verbose_name='Ближайшее метро')
    is_studio = models.IntegerField(default=0, choices=[(0, 'Нет'), (1, 'Да')], verbose_name='Студия')

    # Дополнительные (заполняются автоматически)
    predicted_price = models.FloatField(verbose_name='Предсказанная цена', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Квартира {self.total_meters}м², {self.rooms_count}к, цена: {self.predicted_price} млн руб"