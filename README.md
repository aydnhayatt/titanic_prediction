# 🚢 Titanic Hayatta Kalma Tahmin Projesi

Bu proje, Titanic yolcularının hayatta kalıp kalamayacağını tahmin etmek için geliştirilmiş bir makine öğrenmesi uygulamasıdır.  
Model olarak **Random Forest** sınıflandırıcısı kullanılmıştır ve veri ön işleme, özellik mühendisliği adımları gerçekleştirilmiştir.

## 🧠Proje Özellikleri

- Eksik verilerin doldurulması (Yaş, Kabin bilgisi, Biniş limanı vb.)  
- Yolcu isimlerinden unvan çıkarımı ve nadir unvanların gruplanması  
- Yeni özellikler: Aile büyüklüğü, yalnız olma durumu, kabin bilgisi  
- Model olarak Random Forest sınıflandırıcısı  
- Doğrulama seti üzerinde %82-83 civarında başarı  
- Kullanıcı dostu **Gradio** arayüzü ile tahmin yapabilme imkanı
-- Veri Kümesi: [Kaggle - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)



## 📸 Arayüz Ekran Görüntüsü
![Gradio UI](screenshots/titanic_ui.png)



## 🎯 Kullanılan Teknolojiler

- Python
- Pandas & Scikit-learn
- Random Forest Classifier
- Gradio (Web arayüzü)
- Matplotlib (Görselleştirme)



## 🚀 Başlatmak için

### Gerekli kütüphaneler

```bash
pip install pandas numpy scikit-learn matplotlib gradio joblib

## Nasıl Çalıştırılır?
bash
Kopyala
Düzenle
python main.py         # Modeli eğitip .pkl dosyasını oluşturur
python gradio_app.py   # Gradio arayüzünü başlatır

## Dosya Yapısı
titanic_tahmin/
├── main.py                # Model eğitimi ve kaydetme
├── gradio_app.py          # Gradio arayüz kodu
├── titanic_rf_model.pkl   # Eğitilen model dosyası
├── train.csv              # Eğitim verisi
├── test.csv               # Test verisi
├── screenshots/
│   └── titanic_ui.png     # Arayüz ekran görüntüsü
├── README.md              # Bu dosya


## Geliştirici

Hayat Aydın 👩‍💻  
[LinkedIn](https://www.linkedin.com/in/hayataydin) | [GitHub](https://github.com/aydnhayatt)