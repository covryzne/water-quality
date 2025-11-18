# Water Quality Dashboard

Dashboard ini menampilkan analisis dan prediksi kualitas air menggunakan Streamlit.

## Cara Menjalankan Secara Lokal

### 1. Clone Repository
```
git clone <repo-url>
cd water-quality/Streamlit
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)
```
python -m venv venv
# Aktifkan venv (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Jalankan Streamlit
```
streamlit run dashboard.py
```

### 5. Buka di Browser
Akses [http://localhost:8501](http://localhost:8501) untuk melihat dashboard.

## Catatan Penting
- Pastikan file dataset (`panelA_cleaned.csv`, `panelB_cleaned.csv`) ada di folder `../Dataset`.
- Pastikan file model (`.pkl`) ada di folder `../Models`.
- Jika menggunakan fitur AI Gemini, set environment variable `GEMINI_API_KEY` di `.env`.
- Disarankan menggunakan Python 3.10 atau lebih baru.

---

Jika ada error terkait versi Python atau library, upgrade Python ke versi terbaru dan install ulang dependensi.
