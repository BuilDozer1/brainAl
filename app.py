import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# --- YAPILANDIRMA ---
MODEL_PATH = 'NeuroAI_Brain_Tumor_Model.h5'
# Modelin eğitildiği giriş boyutu (EfficientNet için genelde 224 veya 240'tır)
IMG_SIZE = (224, 224) 

# Sınıf isimleri (HTML'deki CLASSES dizisi ile aynı sırada olmalı)
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Modeli yükle
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print(f"✅ Model başarıyla yüklendi: {MODEL_PATH}")
else:
    print(f"❌ HATA: {MODEL_PATH} dosyası bulunamadı!")

def preprocess_image(img_path):
    """Görüntüyü yükler ve model için hazır hale getirir."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # EfficientNet modelleri genellikle dahili normalizasyona sahiptir, 
    # ancak modelin eğitim şekline göre /255.0 gerekebilir.
    img_array /= 255.0 
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400

    try:
        # Geçici olarak dosyayı kaydet
        upload_path = 'temp_upload.jpg'
        file.save(upload_path)

        # Görüntü işleme ve Tahmin
        processed_img = preprocess_image(upload_path)
        prediction = model.predict(processed_img)[0]

        # Sonuçları formatla (JS'nin beklediği yapı: {isim: ..., oran: ...})
        results = []
        for i in range(len(CLASS_NAMES)):
            results.append({
                'isim': CLASS_NAMES[i],
                'oran': round(float(prediction[i]) * 100, 2)
            })

        # Orana göre yüksekten düşüğe sırala
        results = sorted(results, key=lambda x: x['oran'], reverse=True)

        # Geçici dosyayı sil
        os.remove(upload_path)

        return jsonify({'sonuclar': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 'templates' klasörü yoksa oluştur uyarısı
    if not os.path.exists('templates'):
        print("⚠️ 'templates' klasörü bulunamadı. index.html dosyasını bu klasöre koyun.")
    
    app.run(host='0.0.0.0', port=8080)
