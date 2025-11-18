import streamlit as st
import google.generativeai as genai
import requests
import pandas as pd
import os
import numpy as np
import joblib
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'Dataset')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'Models')
PANELA_CSV = os.path.join(DATASET_DIR, 'panelA_cleaned.csv')
PANELB_CSV = os.path.join(DATASET_DIR, 'panelB_cleaned.csv')
MODEL_PATH_A = os.path.join(MODELS_DIR, 'best_model_rf_panelA.pkl')
SCALER_PATH_A = os.path.join(MODELS_DIR, 'scaler_panelA.pkl')
ENCODER_PATH_A = os.path.join(MODELS_DIR, 'label_encoder_panelA.pkl')
MODEL_PATH_B = os.path.join(MODELS_DIR, 'best_model_rf_panelB.pkl')
SCALER_PATH_B = os.path.join(MODELS_DIR, 'scaler_panelB.pkl')
ENCODER_PATH_B = os.path.join(MODELS_DIR, 'label_encoder_panelB.pkl')

try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Gemini init error: {e}")


def get_water_quality_recommendation(sensor_values, label):
    """
    Memanggil Gemini API untuk interpretasi dan rekomendasi kualitas air.
    sensor_values: dict, label: str
    """
    prompt = f"""
    Sensor:
    - flow1: {sensor_values.get('flow1', '-')}
    - turbidity: {sensor_values.get('turbidity', '-')}
    - tds: {sensor_values.get('tds', '-')}
    - ph: {sensor_values.get('ph', '-')}
    - flow2: {sensor_values.get('flow2', '-')}
    Prediksi kualitas air: {label}

    Tugas Anda:
    1. Berikan interpretasi kualitas air berdasarkan data di atas.
    2. Berikan rekomendasi treatment atau tindakan yang sesuai.
    Jawab dalam 2 paragraf (interpretasi dan rekomendasi).
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini Error] {e}"
    
st.set_page_config(page_title="Dashboard Kualitas Air", layout="wide")

st.write("Cloud working directory:", os.getcwd())
st.write("List files:", os.listdir())
st.write("List parent:", os.listdir(".."))
st.write("List repo root:", os.listdir(".."))
st.write("BASE_DIR:", BASE_DIR)
st.write("DATASET_DIR:", DATASET_DIR)
st.write("Dataset exists:", os.path.exists(DATASET_DIR))
st.write("PANELA exists:", os.path.exists(PANELA_CSV))


# LOAD & PREP DATA SEKALI, CACHE
@st.cache_data
def load_and_prepare_data():
    panelA = pd.read_csv(PANELA_CSV)
    panelB = pd.read_csv(PANELB_CSV)
    return panelA, panelB

panelA, panelB = load_and_prepare_data()

# Sidebar navigation
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ("Data & Info", "Model & Evaluasi", "Simulasi Prediksi")
)

# MENU: DATA & INFO
if menu == "Data & Info":
    st.title("Data & Info")

    tab1, tab2 = st.tabs(["Panel A (Sebelum Filtrasi)", "Panel B (Setelah Filtrasi)"])

    label_order = ["Cokelat", "Orange", "Biru", "Putih"]
    label_colors = {"Cokelat": "#8B4513", "Orange": "#FFA500", "Biru": "#1E90FF", "Putih": "#B0B0B0"}

    # Panel A
    with tab1:
        st.subheader("Ringkasan Dataset Panel A")
        date_col_A = "createdAt"
        # Tidak ada filter tanggal, tampilkan semua data
        panelA[date_col_A] = pd.to_datetime(panelA[date_col_A], errors='coerce')
        # Convert to string to avoid Arrow serialization error
        panelA[date_col_A] = panelA[date_col_A].astype(str)
        filtered_A = panelA.copy()

        # ===== Metrics =====
        with st.container():
            cols = st.columns(4)
            cols[0].metric("N Data", f"{len(filtered_A):,}")
            cols[1].metric("PH Mean", f"{filtered_A['ph'].mean():.2f}")
            cols[2].metric("Turbidity Mean", f"{filtered_A['turbidity'].mean():.2f}")
            cols[3].metric("TDS Mean", f"{filtered_A['tds'].mean():.2f}")

        # ===== Show Table =====
        df_show_A = filtered_A.copy()
        # Convert all datetime, timedelta, and problematic object columns to string
        for col in df_show_A.columns:
            if pd.api.types.is_datetime64_any_dtype(df_show_A[col]) or pd.api.types.is_timedelta64_dtype(df_show_A[col]):
                df_show_A[col] = df_show_A[col].astype(str)
            elif df_show_A[col].dtype == object:
                # If object column contains Timestamp, convert to string
                if df_show_A[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                    df_show_A[col] = df_show_A[col].astype(str)
        st.dataframe(df_show_A.head(), width='stretch')

        # ===== Pie Chart =====
        st.write("Distribusi Label Kualitas Air")

        label_df_A = (
            filtered_A['quality_label']
            .value_counts()
            .reindex(label_order)
            .reset_index()
        )
        label_df_A.columns = ['quality_label', 'count']

        pie_A = alt.Chart(label_df_A).mark_arc(innerRadius=40).encode(
            theta='count:Q',
            color=alt.Color('quality_label:N',
                            scale=alt.Scale(domain=label_order,
                                            range=[label_colors[x] for x in label_order]),
                            legend=None),
            tooltip=['quality_label', 'count']
        ).properties(height=300, width=300)

        chart_col, legend_col = st.columns([3,1])
        with legend_col:
            for label in label_order:
                st.markdown(
                    f'<div style="display:flex;align-items:center;margin-bottom:8px;">'
                    f'<div style="width:18px;height:18px;background:{label_colors[label]};border-radius:3px;margin-right:8px;"></div>'
                    f'<span>{label}</span></div>',
                    unsafe_allow_html=True
                )
        with chart_col:
            st.altair_chart(pie_A)

        with st.expander("Statistik Lengkap Panel A"):
            st.dataframe(filtered_A.describe(), width='stretch')

    # Panel B
    with tab2:
        st.subheader("Ringkasan Dataset Panel B")
        date_col_B = "createdAt"
        # Tidak ada filter tanggal, tampilkan semua data
        panelB[date_col_B] = pd.to_datetime(panelB[date_col_B], errors='coerce')
        # Convert to string to avoid Arrow serialization error
        panelB[date_col_B] = panelB[date_col_B].astype(str)
        filtered_B = panelB.copy()

        # Metrics
        with st.container():
            cols = st.columns(4)
            cols[0].metric("N Data", f"{len(filtered_B):,}")
            cols[1].metric("PH Mean", f"{filtered_B['ph'].mean():.2f}")
            cols[2].metric("Turbidity Mean", f"{filtered_B['turbidity'].mean():.2f}")
            cols[3].metric("TDS Mean", f"{filtered_B['tds'].mean():.2f}")

        # Show Table
        df_show_B = filtered_B.copy()
        # Convert all datetime, timedelta, and problematic object columns to string
        for col in df_show_B.columns:
            if pd.api.types.is_datetime64_any_dtype(df_show_B[col]) or pd.api.types.is_timedelta64_dtype(df_show_B[col]):
                df_show_B[col] = df_show_B[col].astype(str)
            elif df_show_B[col].dtype == object:
                if df_show_B[col].apply(lambda x: isinstance(x, pd.Timestamp)).any():
                    df_show_B[col] = df_show_B[col].astype(str)
        st.dataframe(df_show_B.head(), width='stretch')

        # Pie Chart
        st.write("Distribusi Label Kualitas Air")

        label_df_B = (
            filtered_B['quality_label']
            .value_counts()
            .reindex(label_order)
            .reset_index()
        )
        label_df_B.columns = ['quality_label', 'count']

        pie_B = alt.Chart(label_df_B).mark_arc(innerRadius=40).encode(
            theta='count:Q',
            color=alt.Color('quality_label:N',
                            scale=alt.Scale(domain=label_order,
                                            range=[label_colors[x] for x in label_order]),
                            legend=None),
            tooltip=['quality_label', 'count']
        ).properties(height=300, width=300)

        chart_col, legend_col = st.columns([3,1])
        with legend_col:
            for label in label_order:
                st.markdown(
                    f'<div style="display:flex;align-items:center;margin-bottom:8px;">'
                    f'<div style="width:18px;height:18px;background:{label_colors[label]};border-radius:3px;margin-right:8px;"></div>'
                    f'<span>{label}</span></div>',
                    unsafe_allow_html=True
                )
        with chart_col:
            st.altair_chart(pie_B)

        with st.expander("Statistik Lengkap Panel B"):
            st.dataframe(filtered_B.describe(), width='stretch')


elif menu == "Model & Evaluasi":
    st.title("Model & Evaluasi")
    st.subheader("Perbandingan Akurasi Model Panel A vs Panel B")
    acc_data = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"],
        "Panel A": [0.9590, 0.9618, 1.0, 0.8530],
        "Panel B": [0.9418, 0.9879, 0.9999, 0.1476]
    }
    df_acc = pd.DataFrame(acc_data)

    # Visualisasi Altair Bar Chart
    df_melt = df_acc.melt(id_vars="Model", var_name="Panel", value_name="Akurasi")
    chart = alt.Chart(df_melt).mark_bar().encode(
        x=alt.X('Model:N', title='Model'),
        xOffset='Panel:N',
        y=alt.Y('Akurasi:Q', title='Akurasi', scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color('Panel:N', scale=alt.Scale(domain=['Panel A', 'Panel B'], range=['royalblue', 'orange'])),
        tooltip=['Model', 'Panel', alt.Tooltip('Akurasi', format='.4f')]
    ).properties(
        width=400,
        height=550,
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Confusion Matrix Per Model (Panel A vs Panel B)")
    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM"
    ]
    model_files = [
        "logreg.png",
        "dt.png",
        "rf.png",
        "svm.png"
    ]
    panelA_dir = os.path.join(BASE_DIR, "Assets", "panelA")
    panelB_dir = os.path.join(BASE_DIR, "Assets", "panelB")

    @st.cache_resource
    def load_image_cached(path):
        return Image.open(path)

    for i, model in enumerate(model_names):
        st.markdown(f"#### {model}")
        colA, colB = st.columns(2)
        with colA:
            st.caption("Panel A")
            img_path_A = os.path.join(panelA_dir, model_files[i])
            if os.path.exists(img_path_A):
                st.image(load_image_cached(img_path_A), width='stretch')
            else:
                st.warning(f"Gambar confusion matrix untuk {model} Panel A tidak ditemukan.")
        with colB:
            st.caption("Panel B")
            img_path_B = os.path.join(panelB_dir, model_files[i])
            if os.path.exists(img_path_B):
                st.image(load_image_cached(img_path_B), width='stretch')
            else:
                st.warning(f"Gambar confusion matrix untuk {model} Panel B tidak ditemukan.")

elif menu == "Simulasi Prediksi":
    st.title("Prediksi Kualitas Air")
    tabA, tabB = st.tabs(["Panel A (Sebelum Filtrasi)", "Panel B (Setelah Filtrasi)"])

    with tabA:
        st.subheader("Prediksi Kualitas Air Panel A")
        col1, col2 = st.columns(2)
        with col1:
            flow1 = st.number_input("flow1 (A)", value=2.5, min_value=0.0, max_value=10.0, step=0.1, key="flow1_A")
            turbidity = st.number_input("turbidity (A)", value=5.0, min_value=0.0, max_value=100.0, step=0.1, key="turbidity_A")
        with col2:
            ph = st.number_input("ph (A)", value=7.2, min_value=0.0, max_value=14.0, step=0.01, key="ph_A")
            tds = st.number_input("tds (A)", value=350, min_value=0, max_value=2000, step=1, key="tds_A")

        if st.button("Prediksi Panel A"):
            model = joblib.load(MODEL_PATH_A)
            scaler = joblib.load(SCALER_PATH_A)
            le = joblib.load(ENCODER_PATH_A)
            input_df = pd.DataFrame([[ph, turbidity, tds, flow1]], columns=["ph", "turbidity", "tds", "flow1"])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)
            label = le.inverse_transform(pred)[0]
            st.success(f"Prediksi kualitas air Panel A: **{label}**")

            # LLM Recommendation
            sensor_values = {"ph": ph, "turbidity": turbidity, "tds": tds, "flow1": flow1}
            with st.spinner("Mengambil rekomendasi dari AI..."):
                llm_result = get_water_quality_recommendation(sensor_values, label)
            st.info(llm_result)

    with tabB:
        st.subheader("Prediksi Kualitas Air Panel B")
        col1, col2 = st.columns(2)
        with col1:
            flow1 = st.number_input("flow1 (B)", value=2.7, min_value=0.0, max_value=10.0, step=0.1, key="flow1_B")
            turbidity = st.number_input("turbidity (B)", value=1.2, min_value=0.0, max_value=100.0, step=0.1, key="turbidity_B")
            ph = st.number_input("ph (B)", value=7.4, min_value=0.0, max_value=14.0, step=0.01, key="ph_B")
        with col2:
            tds = st.number_input("tds (B)", value=220, min_value=0, max_value=2000, step=1, key="tds_B")
            flow2 = st.number_input("flow2 (B)", value=2.7, min_value=0.0, max_value=10.0, step=0.1, key="flow2_B")

        if st.button("Prediksi Panel B"):
            model = joblib.load(MODEL_PATH_B)
            scaler = joblib.load(SCALER_PATH_B)
            le = joblib.load(ENCODER_PATH_B)
            input_df = pd.DataFrame([[flow1, turbidity, ph, tds, flow2]], columns=["flow1", "turbidity", "ph", "tds", "flow2"])
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)
            label = le.inverse_transform(pred)[0]
            st.success(f"Prediksi kualitas air Panel B: **{label}**")

            # LLM Recommendation
            sensor_values = {"ph": ph, "turbidity": turbidity, "tds": tds, "flow1": flow1, "flow2": flow2}
            with st.spinner("Mengambil rekomendasi dari AI..."):
                llm_result = get_water_quality_recommendation(sensor_values, label)
            st.info(llm_result)