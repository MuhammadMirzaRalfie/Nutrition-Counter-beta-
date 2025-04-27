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
    # Upload file
    upload_url = "https://api.assemblyai.com/v2/upload"
    headers = {'authorization': ASSEMBLYAI_API_KEY}
    upload_response = requests.post(upload_url, headers=headers, files={'file': file})
    
    if upload_response.status_code != 200:
        st.error(f"Gagal upload audio! ({upload_response.status_code}) {upload_response.text}")
        return None

    audio_url = upload_response.json()['upload_url']

    # Request transcription
    transcript_url = "https://api.assemblyai.com/v2/transcript"
    transcript_request = {"audio_url": audio_url, "language_code": "id"}
    transcript_response = requests.post(transcript_url, headers=headers, json=transcript_request)

    if transcript_response.status_code != 200:
        st.error(f"Gagal minta transkrip! ({transcript_response.status_code}) {transcript_response.text}")
        return None

    transcript_id = transcript_response.json()['id']

    # Polling untuk hasil
    polling_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    while True:
        poll_response = requests.get(polling_url, headers=headers)
        status = poll_response.json()['status']

        if status == 'completed':
            return poll_response.json()['text']
        elif status == 'failed':
            st.error("Transkripsi gagal!")
            return None
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
