from django.shortcuts import render, redirect, get_object_or_404
from .models import PropertyEvaluation
from .forms import PropertyEvaluationForm
from datetime import datetime


def index(request):
    return render(request, 'property_evaluation/index.html')


def evaluate_property(request):
    if request.method == 'POST':
        form = PropertyEvaluationForm(request.POST)
        if form.is_valid():
            property_data = form.save(commit=False)

            # Расчёт возраста дома
            current_year = datetime.now().year
            year_built = property_data.year_built or 2000
            building_age = current_year - year_built

            # Подготовка признаков для модели (6 штук)
            features = {
                'total_meters': float(property_data.total_area),
                'kitchen_meters': float(property_data.kitchen_area or 0),
                'rooms_count': int(property_data.rooms_count),
                'floor': int(property_data.floor),
                'floors_count': int(property_data.total_floors),
                'building_age': building_age,
            }

            # Импортируем предсказатель
            from .ml_utils import predictor

            predicted_price = predictor.predict(features)

            if predicted_price:
                property_data.predicted_price = predicted_price
                property_data.save()
                return redirect('result', pk=property_data.pk)
            else:
                property_data.predicted_price = property_data.total_area * 80000
                property_data.save()
                return redirect('result', pk=property_data.pk)

    else:
        form = PropertyEvaluationForm()

    return render(request, 'property_evaluation/evaluate.html', {'form': form})


def result(request, pk):
    property_data = get_object_or_404(PropertyEvaluation, pk=pk)
    return render(request, 'property_evaluation/result.html', {'property': property_data})