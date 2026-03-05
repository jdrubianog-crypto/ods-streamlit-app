import streamlit as st
import joblib
import os
import urllib.request

st.set_page_config(page_title="Clasificador ODS", page_icon="🌍", layout="centered")
st.title("🌍 Clasificador de textos según ODS")
st.write(
    "La aplicación descarga un modelo previamente entrenado desde Google Drive "
    "y clasifica un texto en 1 de los 17 ODS."
)

# ==========================================================
# 1) CONFIGURACIÓN: pegar el ID del archivo en Google Drive
# ==========================================================
# En Drive, el enlace suele verse así:
# https://drive.google.com/file/d/ID_DEL_ARCHIVO/view?usp=sharing
# El ID es lo que va entre /d/ y /view
FILE_ID = "1o5yCDBPL1_K6LWa8jisEU9f97mLhXyaR"
MODEL_PATH = "modelo_final_tfidf_svd_lr.joblib"

@st.cache_resource
def load_model():
    # Si el modelo no está en el contenedor, se descarga una vez
    if not os.path.exists(MODEL_PATH):
        st.info("Descargando modelo desde Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
        urllib.request.urlretrieve(url, MODEL_PATH)

    # Cargar el pipeline completo (TF-IDF + SVD + clasificador)
    return joblib.load(MODEL_PATH)

# Carga del modelo (queda cacheado por sesión)
model = load_model()

texto = st.text_area(
    "Ingrese un texto en español",
    height=180,
    placeholder="Pegue aquí un fragmento de texto para clasificarlo según los ODS..."
)

btn = st.button("Clasificar", use_container_width=True)

if btn:
    if texto.strip() == "":
        st.warning("Se requiere un texto no vacío para realizar la predicción.")
    else:
        pred = model.predict([texto])[0]
        st.success(f"ODS predicho: {int(pred)}")

        # Mostrar Top 3 si el pipeline soporta probabilidades
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([texto])[0]
            top3 = proba.argsort()[::-1][:3]
            st.write("Top 3 ODS sugeridos (probabilidad):")
            for i in top3:
                st.write(f"ODS {int(i)} → {float(proba[i]):.4f}")
