import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Streamlit UI
st.title('Sistem Prediksi Promosi Jabatan Pelindo')
st.write('Aplikasi untuk memprediksi potensi promosi karyawan')

# Sidebar untuk unggah file CSV
st.sidebar.header('Unggah Data')
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Fungsi untuk membaca data
def load_data(file):
    return pd.read_csv(file)

# Gunakan data dari file CSV jika diunggah, jika tidak gunakan data sintetis
if uploaded_file is not None:
    data = load_data(uploaded_file)
else:
    def generate_data():
        np.random.seed(42)
        data = {
            'usia': np.random.randint(22, 55, 1000),
            'pendidikan': np.random.choice([1, 2, 3], 1000, p=[0.2, 0.6, 0.2]),
            'kinerja': np.random.normal(75, 10, 1000),
            'pengalaman': np.random.randint(1, 20, 1000),
            'pelatihan': np.random.randint(0, 100, 1000),
            'proyek': np.random.randint(0, 15, 1000),
            'rekomendasi': np.random.choice([0, 1], 1000, p=[0.7, 0.3])
        }
        return pd.DataFrame(data)

    data = generate_data()

# Fungsi untuk melatih model
def train_model(data):
    X = data.drop('Rekomendasi', axis=1)
    y = data['Rekomendasi']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler, accuracy

# Latih model dengan data yang digunakan
model, scaler, accuracy = train_model(data)

# Sidebar parameter input
st.sidebar.header('Parameter Input')

def user_input():
    usia = st.sidebar.slider('Usia', 20, 60, 30)
    pendidikan = st.sidebar.selectbox('Tingkat Pendidikan', ('S1', 'S2', 'S3'))
    kinerja = st.sidebar.slider('Nilai Kinerja', 0.0, 100.0, 75.0)
    pengalaman = st.sidebar.slider('Pengalaman Kerja (tahun)', 0, 30, 5)
    pelatihan = st.sidebar.slider('Jam Pelatihan', 0, 200, 50)
    proyek = st.sidebar.slider('Jumlah Proyek', 0, 30, 5)
    
    pendidikan_map = {'S1': 1, 'S2': 2, 'S3': 3}
    
    return {
        'usia': usia,
        'pendidikan': pendidikan_map[pendidikan],
        'kinerja': kinerja,
        'pengalaman': pengalaman,
        'pelatihan': pelatihan,
        'proyek': proyek
    }

input_data = user_input()

# Tampilkan akurasi model
st.subheader('Akurasi Model')
st.write(f'Akurasi model: {accuracy:.2f}')

# Lakukan prediksi
input_df = pd.DataFrame([input_data])
scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

# Tampilkan hasil prediksi
st.subheader('Hasil Prediksi')
rekomendasi = 'Direkomendasikan' if prediction[0] == 1 else 'Tidak Direkomendasikan'
warna = 'green' if prediction[0] == 1 else 'red'

st.markdown(f"**Hasil Prediksi:** <span style='color:{warna}'>{rekomendasi}</span>", unsafe_allow_html=True)
st.write(f'Probabilitas: {prediction_proba[0][1]:.2f}')

# Feature importance
st.subheader('Faktor Penting dalam Prediksi')
importances = model.feature_importances_
feature_names = data.drop('rekomendasi', axis=1).columns
importance_df = pd.DataFrame({'Fitur': feature_names, 'Penting': importances})
importance_df = importance_df.sort_values('Penting', ascending=False)
st.bar_chart(importance_df.set_index('Fitur'))

# Data preview
st.subheader('Preview Data')
st.write(data.head())


