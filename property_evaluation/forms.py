from django import forms
from .models import PropertyEvaluation

class PropertyEvaluationForm(forms.ModelForm):
    class Meta:
        model = PropertyEvaluation
        fields = [
            'property_type', 'total_area', 'living_area', 'kitchen_area',
            'rooms_count', 'floor', 'total_floors', 'year_built',
            'city', 'district', 'street', 'condition',
            'has_balcony', 'has_parking', 'distance_to_metro'
        ]
        widgets = {
            'property_type': forms.Select(attrs={'class': 'form-control'}),
            'total_area': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'living_area': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'kitchen_area': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'rooms_count': forms.NumberInput(attrs={'class': 'form-control'}),
            'floor': forms.NumberInput(attrs={'class': 'form-control'}),
            'total_floors': forms.NumberInput(attrs={'class': 'form-control'}),
            'year_built': forms.NumberInput(attrs={'class': 'form-control'}),
            'city': forms.TextInput(attrs={'class': 'form-control'}),
            'district': forms.TextInput(attrs={'class': 'form-control'}),
            'street': forms.TextInput(attrs={'class': 'form-control'}),
            'condition': forms.Select(attrs={'class': 'form-control'}),
            'has_balcony': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'has_parking': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
            'distance_to_metro': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
        }
        labels = {
            'property_type': 'Тип недвижимости',
            'total_area': 'Общая площадь (м²)',
            'living_area': 'Жилая площадь (м²)',
            'kitchen_area': 'Площадь кухни (м²)',
            'rooms_count': 'Количество комнат',
            'floor': 'Этаж',
            'total_floors': 'Всего этажей',
            'year_built': 'Год постройки',
            'city': 'Город',
            'district': 'Район',
            'street': 'Улица',
            'condition': 'Состояние',
            'has_balcony': 'Балкон',
            'has_parking': 'Парковка',
            'distance_to_metro': 'Расстояние до метро (км)',
        }