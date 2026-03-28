# Temel imaj olarak Python 3.9 kullanıyoruz (TensorFlow ile çok uyumludur)
FROM python:3.9

# Hugging Face Spaces için standart bir kullanıcı (user) oluşturuyoruz
# Bu adım izin hataları (permission denied) almamak için önemlidir
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Çalışma dizinini belirliyoruz
WORKDIR /app

# Önce sadece requirements.txt dosyasını kopyalayıp kütüphaneleri kuruyoruz
# (Bu sayede kodda değişiklik yapsan bile kütüphaneler baştan indirilmez, hızlı build olur)
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Şimdi app.py, model dosyası, templates ve diğer tüm dosyaları kopyalıyoruz
COPY --chown=user . /app

# Hugging Face'in beklediği 7860 portunu dışarı açıyoruz
EXPOSE 7860

# Uygulamayı başlatacak komut
CMD ["python", "app.py"]