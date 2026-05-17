from django import forms
from .models import PropertyEvaluation


class PropertyEvaluationForm(forms.ModelForm):
    class Meta:
        model = PropertyEvaluation
        fields = [
            'total_meters', 'kitchen_meters', 'rooms_count',
            'floor', 'floors_count', 'year_built',
            'author_type', 'district', 'street', 'house_number', 'is_studio'
        ]
        widgets = {
            'total_meters': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'placeholder': 'Например: 65'}),
            'kitchen_meters': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'placeholder': 'Например: 10.5'}),
            'rooms_count': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Например: 3'}),
            'floor': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Например: 5'}),
            'floors_count': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Например: 12'}),
            'year_built': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Например: 2015'}),
            'author_type': forms.Select(attrs={'class': 'form-control'}, choices=[('real_estate_agent', 'Агент'), ('owner', 'Собственник'), ('developer', 'Застройщик')]),
            'district': forms.Select(attrs={'class': 'form-control'}, choices=[
                ('Центральный', 'Центральный'), ('Октябрьский', 'Октябрьский'),
                ('Ленинский', 'Ленинский'), ('Кировский', 'Кировский'),
                ('Заельцовский', 'Заельцовский'), ('Дзержинский', 'Дзержинский'),
                ('Калининский', 'Калининский'), ('Железнодорожный', 'Железнодорожный'),
                ('Советский', 'Советский'), ('Первомайский', 'Первомайский'),
            ]),
            'street': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Например: Грибоедова'}),
            'house_number': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Например: 75'}),
            'is_studio': forms.Select(attrs={'class': 'form-control'}, choices=[(0, 'Нет'), (1, 'Да')]),
        }
        labels = {
            'total_meters': 'Общая площадь (м²)',
            'kitchen_meters': 'Площадь кухни (м²)',
            'rooms_count': 'Количество комнат',
            'floor': 'Этаж',
            'floors_count': 'Всего этажей',
            'year_built': 'Год постройки',
            'author_type': 'Тип автора',
            'district': 'Район',
            'street': 'Улица',
            'house_number': 'Номер дома',
            'is_studio': 'Студия',
        }