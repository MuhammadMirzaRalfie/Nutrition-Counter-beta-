import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import time

# === AMBIL API KEY DARI SECRETS ===
OPENROUTER_API_KEY = st.secrets["openrouter_api_key"]
ASSEMBLYAI_API_KEY = st.secrets["assemblyai_api_key"]
GEMINI_MODEL = "google/gemini-pro-vision"
GPT_MODEL = "openai/gpt-3.5-turbo"

# === TEMPLATE PROMPT ===
FOOD_DETECTION_PROMPT = """
Dari gambar ini, identifikasikan semua makanan yang terlihat.

Instruksi:
- Sebutkan nama makanan dan jumlahnya secara terpisah.
- Tulis dalam format daftar bullet seperti:
  - 2 kerupuk
  - 1 ayam goreng
  - 1 piring nasi goreng
  - 3 potong timun
- Jika jumlah tidak pasti, berikan estimasi wajar berdasarkan gambar.
- Tidak perlu deskripsi tambahan, hanya daftar saja.
"""

NUTRITION_CALCULATION_PROMPT = """
Berikut ini adalah daftar makanan beserta jumlahnya:

{text}

Tugasmu:
- Tentukan nilai nutrisi standar untuk **satu** unit makanan (kalori, protein, lemak, karbohidrat).
- Kalikan nilai nutrisi tersebut dengan jumlah makanan yang disebutkan.
- Tampilkan hasil kalkulasi **per item** dan **total keseluruhan** di bagian bawah.
- Format hasil dalam bentuk tabel dengan kolom berikut:
  - Nama Makanan
  - Jumlah
  - Kalori
  - Protein (g)
  - Lemak (g)
  - Karbohidrat (g)
"""

# === FUNGSI UTAMA ===

def encode_image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def openrouter_chat(model, messages):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": messages,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        st.error(f"Error OpenRouter: {response.status_code} - {response.text}")
        return None

def detect_food_from_image(image):
    base64_image = encode_image_to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": FOOD_DETECTION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    return openrouter_chat(GEMINI_MODEL, messages)

def calculate_nutrition(food_list_text):
    messages = [
        {
            "role": "user",
            "content": NUTRITION_CALCULATION_PROMPT.format(text=food_list_text)
        }
    ]
    return openrouter_chat(GPT_MODEL, messages)

def transcribe_audio(file):
    base_url = "https://api.assemblyai.com"
    headers = {
        "authorization": ASSEMBLYAI_API_KEY
    }

    # 1. Upload audio file
    upload_endpoint = base_url + "/v2/upload"
    response = requests.post(upload_endpoint, headers=headers, data=file)
    if response.status_code != 200:
        st.error(f"Upload gagal: {response.text}")
        return "Upload gagal!"

    audio_url = response.json()["upload_url"]

    # 2. Request transcription
    transcript_endpoint = base_url + "/v2/transcript"
    data = {
        "audio_url": audio_url,
        "speech_model": "universal"  # pakai 'universal' default kayak di dokumentasi
    }
    response = requests.post(transcript_endpoint, json=data, headers=headers)
    if response.status_code != 200:
        st.error(f"Request transkrip gagal: {response.text}")
        return "Request transkrip gagal!"

    transcript_id = response.json()["id"]
    polling_endpoint = base_url + "/v2/transcript/" + transcript_id

    # 3. Polling hasil transkrip
    while True:
        poll_response = requests.get(polling_endpoint, headers=headers)
        result = poll_response.json()

        if result['status'] == 'completed':
            return result['text']
        elif result['status'] == 'error':
            st.error(f"Transkripsi gagal: {result['error']}")
            return "Transkripsi gagal!"
        else:
            time.sleep(3)

# === STREAMLIT APP ===

st.title("🍽️ Estimasi Nutrisi dari Gambar, Teks, atau Suara!")

tab1, tab2, tab3 = st.tabs(["📷 Gambar", "📝 Teks Manual", "🎙️ Suara"])

# Tab Gambar
with tab1:
    uploaded_file = st.file_uploader("Upload gambar makanan", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Diupload", use_container_width=True)

        with st.spinner("🔍 Mendeteksi makanan..."):
            detected_food = detect_food_from_image(image)
            if detected_food:
                st.subheader("🍴 Makanan Terdeteksi:")
                st.write(detected_food)

                with st.spinner("📊 Menghitung nutrisi..."):
                    nutrition_info = calculate_nutrition(detected_food)
                    if nutrition_info:
                        st.subheader("📈 Estimasi Nutrisi:")
                        st.write(nutrition_info)

# Tab Teks
with tab2:
    manual_text = st.text_area(
        "Masukkan daftar makanan dan jumlahnya (contoh: 2 telur, 1 nasi goreng, 3 tomat):"
    )
    hitung_button = st.button("Hitung Nutrisi dari Teks")
    
    if hitung_button:
        if manual_text.strip() == "":
            st.warning("⚠️ Masukkan daftar makanan terlebih dahulu.")
        else:
            with st.spinner("📊 Menghitung nutrisi..."):
                nutrition_info = calculate_nutrition(manual_text)
                if nutrition_info:
                    st.subheader("📈 Estimasi Nutrisi:")
                    st.write(nutrition_info)

# Tab Suara
with tab3:
    audio_file = st.file_uploader("Upload file suara (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])
    if audio_file:
        with st.spinner("🎧 Mengubah suara menjadi teks..."):
            transcribed_text = transcribe_audio(audio_file)
            if transcribed_text:
                st.subheader("📝 Teks dari Suara:")
                st.write(transcribed_text)

                with st.spinner("📊 Menghitung nutrisi dari suara..."):
                    nutrition_info = calculate_nutrition(transcribed_text)
                    if nutrition_info:
                        st.subheader("📈 Estimasi Nutrisi:")
                        st.write(nutrition_info)
