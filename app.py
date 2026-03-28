import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# --- YAPILANDIRMA ---
MODEL_PATH = 'NeuroAI_Brain_Tumor_Model.h5'
IMG_SIZE = (224, 224)

CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Modeli yükle
model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(f"✅ Model yüklendi: {MODEL_PATH}")
else:
    print(f"❌ Model bulunamadı!")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model yüklenmedi'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yok'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    try:
        # Railway için güvenli path
        upload_path = '/tmp/temp_upload.jpg'
        file.save(upload_path)

        processed_img = preprocess_image(upload_path)
        prediction = model.predict(processed_img)[0]

        results = []
        for i in range(len(CLASS_NAMES)):
            results.append({
                'isim': CLASS_NAMES[i],
                'oran': round(float(prediction[i]) * 100, 2)
            })

        results = sorted(results, key=lambda x: x['oran'], reverse=True)

        # dosyayı sil
        if os.path.exists(upload_path):
            os.remove(upload_path)

        return jsonify({'sonuclar': results})

    except Exception as e:
        print("HATA:", e)  # logda gör
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
