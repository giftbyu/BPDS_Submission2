import streamlit as st
import pandas as pd
import joblib
import numpy as np
import xgboost as xgb

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Dropout Mahasiswa | Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide"
)

# --- MAPPING DATA ---
marital_status_map = {'Single': 1, 'Married': 2, 'Widower': 3, 'Divorced': 4, 'Facto Union': 5, 'Legally Separated': 6}
gender_map = {'Male': 1, 'Female': 0}
boolean_map = {'Yes': 1, 'No': 0}

# --- FUNGSI UNTUK MEMUAT MODEL & PREPROCESSOR ---
@st.cache_resource
def load_assets():
    try:
        preprocessor = joblib.load('preprocessor.joblib')
        xgb_model = joblib.load('xgb_model.joblib')  # PASTIKAN MODEL DALAM FORMAT .joblib
        return preprocessor, xgb_model
    except Exception as e:
        st.error(f"Error memuat model: {str(e)}")
        return None, None

preprocessor, model = load_assets()

# --- TAMPILAN APLIKASI ---
if model and preprocessor:
    st.title("🎓 Sistem Peringatan Dini Dropout Mahasiswa")
    
    # --- INPUT FORM ---
    col1, col2, col3 = st.columns(3)
    with col1:
        curricular_units_1st_sem_grade = st.number_input('Rata-rata Nilai Semester 1 (0-20)', 0.0, 20.0, 12.0, 0.1)
        admission_grade = st.number_input('Nilai Penerimaan (95-190)', 95.0, 190.0, 125.0, 0.1)
        
    with col2:
        tuition_fees_up_to_date_text = st.selectbox('Uang Kuliah Tepat Waktu?', ['Yes', 'No'])
        scholarship_holder_text = st.selectbox('Penerima Beasiswa?', ['No', 'Yes'])
        
    with col3:
        age_at_enrollment = st.slider('Usia saat Pendaftaran', 17, 70, 20)
        gender_text = st.selectbox('Gender', ['Male', 'Female'])

    if st.button('Analisis Risiko Mahasiswa', type="primary"):
        # Konversi input
        input_data = {
            'Marital_status': [1],  # Default: Single
            'Admission_grade': [admission_grade],
            'Tuition_fees_up_to_date': [boolean_map[tuition_fees_up_to_date_text]],
            'Gender': [gender_map[gender_text]],
            'Scholarship_holder': [boolean_map[scholarship_holder_text]],
            'Age_at_enrollment': [age_at_enrollment],
            'Curricular_units_1st_sem_grade': [curricular_units_1st_sem_grade],
            'Debtor': [0],  # Default: Tidak punya hutang
            'Curricular_units_2nd_sem_grade': [0.0]  # Default
        }
        
        # Buat DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Transformasi & prediksi
        data_transformed = preprocessor.transform(input_df)
        prediction_encoded = model.predict(data_transformed)[0]
        prediction_proba = model.predict_proba(data_transformed)[0]
        
        # Tampilkan hasil
        status_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
        prediction_label = status_map[prediction_encoded]
        dropout_risk = prediction_proba[0] * 100
        
        st.metric("Prediksi Status", prediction_label)
        st.metric("Skor Risiko Dropout", f"{dropout_risk:.1f}%")
        
        if prediction_label == 'Dropout':
            st.error("🚨 REKOMENDASI: Intervensi segera diperlukan!")
        else:
            st.success("✅ REKOMENDASI: Mahasiswa dalam kondisi baik")
