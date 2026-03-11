from django.shortcuts import render, redirect
from .forms import PropertyEvaluationForm
from .models import PropertyEvaluation
from .ml_utils import predictor  # ← ЭТО НОВАЯ СТРОКА (импорт модели)


def index(request):
    """Главная страница с формой для оценки"""
    return render(request, 'property_evaluation/index.html')


def evaluate_property(request):
    """Обработка формы и предсказание цены"""
    if request.method == 'POST':
        form = PropertyEvaluationForm(request.POST)
        if form.is_valid():
            # Сохраняем данные в базу (но пока не коммитим)
            property_data = form.save(commit=False)

            # ===== НОВЫЙ КОД: ВЫЗОВ МОДЕЛИ =====
            # Подготовка признаков для модели
            features = {
                'author_type': 1,  # Пока заглушка, потом закодируем
                'district': 1,  # Пока заглушка, потом закодируем
                'floor': property_data.floor,
                'floors_count': property_data.total_floors,
                'kitchen_meters': property_data.kitchen_area or 0,
                'living_meters': property_data.living_area or 0,
                'rooms_count': property_data.rooms_count,
                'total_meters': property_data.total_area,
                'year_of_construction': property_data.year_built or 2000
            }

            # Получаем предсказание от модели
            predicted_price = predictor.predict(features)

            if predicted_price:
                property_data.predicted_price = predicted_price
            else:
                # Если модель не сработала, ставим тестовую цену
                property_data.predicted_price = property_data.total_area * 80000
            # ===================================

            property_data.save()  # Теперь сохраняем с ценой

            # Перенаправляем на страницу с результатом
            return redirect('result', pk=property_data.pk)
    else:
        form = PropertyEvaluationForm()

    return render(request, 'property_evaluation/evaluate.html', {'form': form})


def result(request, pk):
    """Страница с результатом оценки"""
    property_data = PropertyEvaluation.objects.get(pk=pk)
    return render(request, 'property_evaluation/result.html', {'property': property_data})