from django.shortcuts import render, redirect
from .forms import PropertyEvaluationForm
from .models import PropertyEvaluation


def index(request):
    """Главная страница с формой для оценки"""
    return render(request, 'property_evaluation/index.html')


def evaluate_property(request):
    """Обработка формы и предсказание цены"""
    if request.method == 'POST':
        form = PropertyEvaluationForm(request.POST)
        if form.is_valid():
            # Сохраняем данные в базу
            property_data = form.save(commit=False)

            # Здесь позже будет вызов ML модели
            # Сейчас просто установим тестовую цену
            # property_data.predicted_price = 5000000  # пример

            property_data.save()

            # Перенаправляем на страницу с результатом
            return redirect('result', pk=property_data.pk)
    else:
        form = PropertyEvaluationForm()

    return render(request, 'property_evaluation/evaluate.html', {'form': form})


def result(request, pk):
    """Страница с результатом оценки"""
    property_data = PropertyEvaluation.objects.get(pk=pk)
    return render(request, 'property_evaluation/result.html', {'property': property_data})