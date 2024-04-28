from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
import os

from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from tensorflow.keras.models import save_model

model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
model = load_model(model_path)

"""
model = VGG16(weights='imagenet')
save_model(model, 'vgg16.h5')
"""

def predict(request):
    if request.method == 'GET':
        # GETリクエストによるアクセス時の処理を記述
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        # POSTリクエストによるアクセス時の処理を記述
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                img_file = form.cleaned_data['image']
                img_file = BytesIO(img_file.read())
                img = load_img(img_file, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = img_array.reshape((1, 224, 224, 3))

                img_array = preprocess_input(img_array)
                result = model.predict(img_array)
                top5_prediction = decode_predictions(result, top=5)[0]

                decode_predictions(result)
                
                prediction = [
                {'description': description, 'probability': round(probability * 100, 2)}
                for _, description, probability in top5_prediction
                ]

                img_data = request.POST.get('img_data')
                return render(request, 'home.html', {'form': form, 'prediction': prediction, 'img_data': img_data})
            except Exception as e:
                print(e)
                return render(request, 'home.html', {'form': form, 'error': f'画像処理中にエラーが発生しました: {str(e)}'})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})