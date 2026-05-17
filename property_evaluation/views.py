from django.shortcuts import render, redirect, get_object_or_404
from .models import PropertyEvaluation
from .forms import PropertyEvaluationForm
from datetime import datetime
from .ml_utils import predictor


def index(request):
    return render(request, 'property_evaluation/index.html')


def evaluate_property(request):
    if request.method == 'POST':
        form = PropertyEvaluationForm(request.POST)
        if form.is_valid():
            property_data = form.save(commit=False)

            current_year = datetime.now().year
            year_built = property_data.year_built or 2000
            property_data.house_age = current_year - year_built

            features = {
                'author_type': property_data.author_type,
                'district': property_data.district,
                'floor': property_data.floor,
                'floors_count': property_data.floors_count,
                'kitchen_meters': float(property_data.kitchen_meters),
                'rooms_count': property_data.rooms_count,
                'street': property_data.street or '',
                'total_meters': float(property_data.total_meters),
                'is_studio': int(property_data.is_studio),
                'house_age': property_data.house_age,
                'house_number': property_data.house_number or '',
            }

            predicted_price_rub = predictor.predict(features)

            if predicted_price_rub:
                property_data.predicted_price = predicted_price_rub / 1_000_000
            else:
                property_data.predicted_price = float(property_data.total_meters) * 0.13

            property_data.save()
            return redirect('result', pk=property_data.pk)

        else:
            return render(request, 'property_evaluation/evaluate.html', {'form': form})

    form = PropertyEvaluationForm()
    return render(request, 'property_evaluation/evaluate.html', {'form': form})


def result(request, pk):
    property_data = get_object_or_404(PropertyEvaluation, pk=pk)
    return render(request, 'property_evaluation/result.html', {'property': property_data})