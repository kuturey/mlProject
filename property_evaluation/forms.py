from django import forms
from .models import PropertyEvaluation

class PropertyEvaluationForm(forms.ModelForm):
    class Meta:
        model = PropertyEvaluation
        fields = ['total_area', 'kitchen_area', 'rooms_count', 'floor', 'total_floors', 'year_built']
        widgets = {
            'total_area': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'placeholder': 'Например: 65'}),
            'kitchen_area': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1', 'placeholder': 'Например: 10.5'}),
            'rooms_count': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Например: 3'}),
            'floor': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Например: 5'}),
            'total_floors': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Например: 12'}),
            'year_built': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Например: 2015'}),
        }
        labels = {
            'total_area': 'Общая площадь (м²)',
            'kitchen_area': 'Площадь кухни (м²)',
            'rooms_count': 'Количество комнат',
            'floor': 'Этаж',
            'total_floors': 'Всего этажей',
            'year_built': 'Год постройки',
        }