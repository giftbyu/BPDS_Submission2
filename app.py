# app.py (Versi Definitif dengan Metode Pemuatan Native XGBoost)

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb # Pastikan xgboost di-import

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa | Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide"
)

# --- MAPPING DATA (SAMA SEPERTI SEBELUMNYA) ---
marital_status_map = {'Single': 1, 'Married': 2, 'Widower': 3, 'Divorced': 4, 'Facto Union': 5, 'Legally Separated': 6}
gender_map = {'Male': 1, 'Female': 0}
boolean_map = {'Yes': 1, 'No': 0}

# --- FUNGSI UNTUK MEMUAT ASET MODEL ---
@st.cache_resource
def load_assets():
    try:
        # Muat preprocessor scikit-learn
        preprocessor = joblib.load('preprocessor.joblib')
        
        # Muat model XGBoost sebagai objek Booster inti, bukan wrapper
        booster = xgb.Booster()
        booster.load_model('xgb_model.json')
        
        return preprocessor, booster
    except Exception as e:
        st.error(f"Error memuat file model: {e}. Pastikan 'preprocessor.joblib' dan 'xgb_model.json' ada di repository GitHub Anda.")
        return None, None

preprocessor, booster_model = load_assets()

# --- TAMPILAN APLIKASI (UI TIDAK BERUBAH) ---
if booster_model and preprocessor:
    st.title("🎓 Sistem Peringatan Dini Dropout Mahasiswa")
    st.markdown("Selamat datang di sistem peringatan dini Jaya Jaya Institut. Masukkan data mahasiswa di bawah untuk mendapatkan prediksi status dan skor risiko.")

    st.header("Masukkan Data Mahasiswa")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # ... (Kode input di sini sama persis seperti sebelumnya) ...
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
        # Konversi input teks ke angka (sama seperti sebelumnya)
        tuition_fees_up_to_date = boolean_map[tuition_fees_up_to_date_text]
        scholarship_holder = boolean_map[scholarship_holder_text]
        # ... (kode mapping lainnya sama) ...
        debtor = boolean_map[debtor_text]
        gender = gender_map[gender_text]
        marital_status = marital_status_map[marital_status_text]

        # Penyiapan input DataFrame (sama seperti sebelumnya)
        df_defaults = pd.read_csv('data.csv', delimiter=';')
        cat_features = preprocessor.transformers_[1][2]
        num_features = preprocessor.transformers_[0][2]
        all_features = num_features + cat_features
        input_data = {}
        for col in all_features:
            if col in locals():
                input_data[col] = [locals()[col]]
            else:
                input_data[col] = [df_defaults[col].median()]
        input_df = pd.DataFrame(input_data)
        
        # --- PERUBAHAN UTAMA PADA PROSES PREDIKSI ---
        # 1. Transformasi data input menggunakan preprocessor
        data_transformed = preprocessor.transform(input_df)
        
        # 2. Buat DMatrix, format data native XGBoost
        dmatrix_pred = xgb.DMatrix(data_transformed)
        
        # 3. Lakukan prediksi probabilitas menggunakan Booster
        prediction_proba = booster_model.predict(dmatrix_pred)[0]
        
        # 4. Tentukan kelas prediksi dengan mengambil indeks probabilitas tertinggi
        prediction_encoded = np.argmax(prediction_proba)
        # ----------------------------------------------
        
        status_map_decode = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
        prediction_label = status_map_decode[prediction_encoded]
        
        # Tampilkan hasil (kode tampilan tidak berubah)
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
