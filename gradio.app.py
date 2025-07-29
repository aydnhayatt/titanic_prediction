import gradio as gr
import numpy as np
import joblib

# Kaydedilmiş modeli yükle
model = joblib.load('titanic_rf_model.pkl')

# Tahmin fonksiyonu
def predict_survival(Pclass, Sex, Age, Fare, SibSp, Parch, Embarked_Q, Embarked_S, Has_Cabin, FamilySize, IsAlone):
    input_data = np.array([[Pclass, Sex, Age, Fare, SibSp, Parch, Embarked_Q, Embarked_S, Has_Cabin, FamilySize, IsAlone]])
    prediction = model.predict(input_data)
    return "🟢 Yolcu hayatta kaldı." if prediction[0] == 1 else "🔴 Yolcu hayatta kalamadı."

import gradio as gr

# Arayüz
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 35%, #90caf9 100%) !important;
    min-height: 100vh;
}

.main {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px);
    border-radius: 20px !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    margin: 20px !important;
    padding: 30px !important;
}

h1 {
    background: linear-gradient(135deg, #1976d2, #42a5f5, #64b5f6) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-align: center !important;
    font-size: 2.5rem !important;
    font-weight: bold !important;
    margin-bottom: 20px !important;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1) !important;
}

.description {
    text-align: center !important;
    color: #1565c0 !important;
    font-size: 1.1rem !important;
    margin-bottom: 30px !important;
    font-style: italic !important;
}

.input-container {
    background: rgba(227, 242, 253, 0.7) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin: 10px 0 !important;
    border: 2px solid rgba(25, 118, 210, 0.2) !important;
    transition: all 0.3s ease !important;
}

.input-container:hover {
    background: rgba(227, 242, 253, 0.9) !important;
    border-color: rgba(25, 118, 210, 0.4) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(25, 118, 210, 0.2) !important;
}

label {
    color: #0d47a1 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

.gr-button {
    background: linear-gradient(135deg, #1976d2, #42a5f5) !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 15px 30px !important;
    font-size: 1.1rem !important;
    font-weight: bold !important;
    box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3) !important;
    transition: all 0.3s ease !important;
}

.gr-button:hover {
    background: linear-gradient(135deg, #1565c0, #1976d2) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(25, 118, 210, 0.4) !important;
}

.output {
    background: rgba(227, 242, 253, 0.8) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    border: 2px solid rgba(25, 118, 210, 0.3) !important;
    color: #0d47a1 !important;
    font-size: 1.2rem !important;
    font-weight: bold !important;
    text-align: center !important;
}

.footer {
    text-align: center !important;
    color: #1565c0 !important;
    font-style: italic !important;
    margin-top: 30px !important;
    padding: 20px !important;
    background: rgba(227, 242, 253, 0.5) !important;
    border-radius: 15px !important;
    border: 1px solid rgba(25, 118, 210, 0.2) !important;
}

/* Özel animasyonlar */
@keyframes wave {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.title-emoji {
    animation: wave 2s ease-in-out infinite;
    display: inline-block;
}

/* Input alanları için özel stiller */
.gr-textbox, .gr-number, .gr-radio {
    border-radius: 10px !important;
    border: 2px solid rgba(25, 118, 210, 0.2) !important;
    background: rgba(255, 255, 255, 0.9) !important;
}

.gr-textbox:focus, .gr-number:focus {
    border-color: #1976d2 !important;
    box-shadow: 0 0 10px rgba(25, 118, 210, 0.3) !important;
}
"""

# Arayüz oluşturma
def create_titanic_interface():
    with gr.Blocks(css=custom_css, title="🚢 Titanic Survival Predictor") as iface:
        
        # Başlık ve açıklama
        gr.HTML("""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h1><span class='title-emoji'>🚢</span> Titanic Hayatta Kalma Tahmin Uygulaması <span class='title-emoji'>🚢</span></h1>
            <div class='description'>
                ⚓ Yolcu bilgilerini girerek Titanic kazasında hayatta kalma olasılığını keşfedin ⚓
            </div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color: #1565c0; text-align: center; margin-bottom: 20px;'>👤 Kişisel Bilgiler</h3>")
                
                pclass = gr.Number(
                    label="🎫 Bilet Sınıfı (1 = Birinci Sınıf, 3 = Üçüncü Sınıf)",
                    value=3,
                    minimum=1,
                    maximum=3,
                    elem_classes=["input-container"]
                )
                
                sex = gr.Radio(
                    choices=[("👨 Erkek", 0), ("👩 Kadın", 1)],
                    label="👫 Cinsiyet",
                    value=0,
                    elem_classes=["input-container"]
                )
                
                age = gr.Number(
                    label="🎂 Yaş",
                    value=30,
                    minimum=0,
                    maximum=100,
                    elem_classes=["input-container"]
                )
                
                fare = gr.Number(
                    label="💰 Bilet Ücreti (£)",
                    value=32.0,
                    minimum=0,
                    elem_classes=["input-container"]
                )
            
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color: #1565c0; text-align: center; margin-bottom: 20px;'>👨‍👩‍👧‍👦 Aile Bilgileri</h3>")
                
                sibsp = gr.Number(
                    label="💑 Kardeş/Eş Sayısı",
                    value=0,
                    minimum=0,
                    elem_classes=["input-container"]
                )
                
                parch = gr.Number(
                    label="👶 Anne-Baba/Çocuk Sayısı",
                    value=0,
                    minimum=0,
                    elem_classes=["input-container"]
                )
                
                family_size = gr.Number(
                    label="👨‍👩‍👧‍👦 Toplam Aile Büyüklüğü",
                    value=1,
                    minimum=1,
                    elem_classes=["input-container"]
                )
                
                is_alone = gr.Radio(
                    choices=[("👥 Hayır, ailemle", 0), ("🚶 Evet, yalnızım", 1)],
                    label="🤷 Yalnız mı seyahat ediyorsunuz?",
                    value=1,
                    elem_classes=["input-container"]
                )
            
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color: #1565c0; text-align: center; margin-bottom: 20px;'>🚢 Seyahat Detayları</h3>")
                
                embarked_q = gr.Radio(
                    choices=[("❌ Hayır", 0), ("✅ Evet", 1)],
                    label="🇮🇪 Queenstown'dan mı bindiz?",
                    value=0,
                    elem_classes=["input-container"]
                )
                
                embarked_s = gr.Radio(
                    choices=[("❌ Hayır", 0), ("✅ Evet", 1)],
                    label="🇬🇧 Southampton'dan mı bindiniz?",
                    value=1,
                    elem_classes=["input-container"]
                )
                
                has_cabin = gr.Radio(
                    choices=[("❌ Hayır", 0), ("✅ Evet", 1)],
                    label="🏠 Özel kabin var mı?",
                    value=0,
                    elem_classes=["input-container"]
                )
        
        # Tahmin butonu
        predict_btn = gr.Button(
            "🔮 Hayatta Kalma Olasılığını Hesapla",
            elem_classes=["gr-button"],
            size="lg"
        )
        
        # Sonuç alanı
        output = gr.Textbox(
            label="📊 Tahmin Sonucu",
            placeholder="Tahmin sonucu burada görünecek...",
            elem_classes=["output"],
            lines=3
        )
        
        # Alt bilgi
        gr.HTML("""
        <div class='footer'>
            <p> <strong>🚢 Titanic Project | Developed by Hayat Aydın👩‍💻</strong> </p>
            <p style='font-size: 0.9rem; margin-top: 10px;'>
                🌊 "Yaşam bir yolculuktur, her anı değerlidir" 🌊
            </p>
        </div>
        """)
        
        # Buton tıklanınca çalışacak fonksiyon
        predict_btn.click(
            fn=predict_survival,
            inputs=[pclass, sex, age, fare, sibsp, parch, embarked_q, embarked_s, has_cabin, family_size, is_alone],
            outputs=output
        )
    
    return iface

# Arayüzü başlatma
iface = create_titanic_interface()
iface.launch(
    share=True,
    inbrowser=True,
    favicon_path=None,
    show_error=True
)
