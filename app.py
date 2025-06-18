# app.py (Versi Final dengan Perbaikan Error dan Label yang Mudah Dibaca)

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa | Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide"
)

# --- MAPPING DATA SESUAI README.MD ---
marital_status_map = {'Single': 1, 'Married': 2, 'Widower': 3, 'Divorced': 4, 'Facto Union': 5, 'Legally Separated': 6}
gender_map = {'Male': 1, 'Female': 0}
boolean_map = {'Yes': 1, 'No': 0}

# --- FUNGSI UNTUK MEMUAT MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('final_dropout_prediction_model.joblib')
        return model
    except FileNotFoundError:
        st.error("File model 'final_dropout_prediction_model.joblib' tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py.")
        return None

model = load_model()

# --- TAMPILAN APLIKASI ---
if model:
    st.title("🎓 Sistem Peringatan Dini Dropout Mahasiswa")
    st.markdown("Selamat datang di sistem peringatan dini Jaya Jaya Institut. Masukkan data mahasiswa di bawah untuk mendapatkan prediksi status dan skor risiko.")

    # --- INPUT DARI PENGGUNA DENGAN LABEL TEKS ---
    st.header("Masukkan Data Mahasiswa")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Data Akademik")
        curricular_units_1st_sem_grade = st.number_input('Rata-rata Nilai Semester 1 (0-20)', 0.0, 20.0, 12.0, 0.1)
        curricular_units_2nd_sem_grade = st.number_input('Rata-rata Nilai Semester 2 (0-20)', 0.0, 20.0, 12.0, 0.1)
        admission_grade = st.number_input('Nilai Penerimaan (95-190)', 95.0, 190.0, 125.0, 0.1)

    with col2:
        st.subheader("Data Finansial")
        tuition_fees_up_to_date_text = st.selectbox('Uang Kuliah Tepat Waktu?', ['Yes', 'No'])
        scholarship_holder_text = st.selectbox('Penerima Beasiswa?', ['No', 'Yes'])
        debtor_text = st.selectbox('Memiliki Hutang?', ['No', 'Yes'])

    with col3:
        st.subheader("Data Personal")
        age_at_enrollment = st.slider('Usia saat Pendaftaran', 17, 70, 20)
        gender_text = st.selectbox('Gender', ['Male', 'Female'])
        marital_status_text = st.selectbox('Status Pernikahan', list(marital_status_map.keys()))

    # Tombol Prediksi
    if st.button('Analisis Risiko Mahasiswa', type="primary", use_container_width=True):
        # --- KONVERSI INPUT TEKS KE ANGKA UNTUK MODEL ---
        tuition_fees_up_to_date = boolean_map[tuition_fees_up_to_date_text]
        scholarship_holder = boolean_map[scholarship_holder_text]
        debtor = boolean_map[debtor_text]
        gender = gender_map[gender_text]
        marital_status = marital_status_map[marital_status_text]
        
        # --- PERBAIKAN LOGIKA PENYIAPAN DATA ---
        # 1. Muat data asli untuk mendapatkan daftar kolom & nilai default
        df_defaults = pd.read_csv('data.csv', delimiter=';')

        # 2. <-- INI BAGIAN YANG DIPERBAIKI -->
        # Cek apakah kolom 'Status' ada, jika ada, buang. Jika tidak, gunakan apa adanya.
        if 'Status' in df_defaults.columns:
            feature_columns = df_defaults.drop('Status', axis=1).columns
        else:
            feature_columns = df_defaults.columns

        # 3. Buat dictionary untuk input
        input_data = {}
        for col in feature_columns:
            if col in locals():
                input_data[col] = [locals()[col]]
            else:
                input_data[col] = [df_defaults[col].median()]
        
        input_df = pd.DataFrame(input_data)

        # Lakukan prediksi
        prediction_encoded = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        status_map_decode = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
        prediction_label = status_map_decode[prediction_encoded]
        
        # --- TAMPILKAN HASIL PREDIKSI ---
        st.header("Hasil Analisis")
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            if prediction_label == 'Dropout':
                st.metric(label="**Prediksi Status**", value=prediction_label, delta=f"{prediction_proba[0]:.0%} Confidence", delta_color="inverse")
            elif prediction_label == 'Enrolled':
                st.metric(label="**Prediksi Status**", value=prediction_label, delta_color="off")
            else:
                st.metric(label="**Prediksi Status**", value=prediction_label, delta=f"{prediction_proba[2]:.0%} Confidence", delta_color="normal")
        
        with res_col2:
            dropout_risk = prediction_proba[0] * 100
            st.metric(label="**Skor Risiko Dropout**", value=f"{dropout_risk:.2f}%")

        if prediction_label == 'Dropout':
            st.error("REKOMENDASI: Mahasiswa ini menunjukkan risiko tinggi untuk dropout. Disarankan untuk segera memberikan bimbingan dan konseling khusus.", icon="🚨")
        else:
            st.success("REKOMENDASI: Mahasiswa ini berada di jalur yang aman. Lanjutkan pemantauan reguler.", icon="✅")