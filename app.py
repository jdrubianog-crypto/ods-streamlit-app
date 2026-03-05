
import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Clasificador ODS", page_icon="🌍", layout="centered")
st.title("🌍 Clasificador de textos según ODS")
st.write(
    "La aplicación carga un modelo previamente entrenado y clasifica un texto en 1 de los 17 ODS. "
    "El modelo utiliza TF-IDF+SVD o Word2Vec, según el artefacto disponible."
)

@st.cache_resource
def load_artifacts():
    # Caso 1: TF-IDF (pipeline completo)
    if os.path.exists("modelo_final_tfidf_svd_lr.joblib"):
        pipe = joblib.load("modelo_final_tfidf_svd_lr.joblib")
        return {"mode": "tfidf", "pipe": pipe}

    # Caso 2: Word2Vec (clasificador + w2v)
    try:
        from gensim.models import Word2Vec
        clf = joblib.load("modelo_final_w2v_lr.joblib")
        w2v = Word2Vec.load("w2v_entrenado.model")
        return {"mode": "w2v", "clf": clf, "w2v": w2v}
    except Exception as e:
        return {"mode": "none", "error": str(e)}

artifacts = load_artifacts()

if artifacts["mode"] == "none":
    st.error("No fue posible cargar los artefactos del modelo. Verifique que los archivos estén en el repositorio.")
    st.code(artifacts.get("error", "Error no especificado"))
    st.stop()

texto = st.text_area(
    "Ingrese un texto en español",
    height=180,
    placeholder="Pegue aquí un fragmento de texto para clasificarlo según los ODS..."
)

col1, col2 = st.columns(2)
with col1:
    btn = st.button("Clasificar", use_container_width=True)
with col2:
    limpiar = st.button("Limpiar", use_container_width=True)

if limpiar:
    st.experimental_rerun()

def tokenize_basic(t: str):
    return str(t).lower().split()

def text_to_vec(tokens, w2v_model):
    vectors = []
    for tok in tokens:
        if tok in w2v_model.wv:
            vectors.append(w2v_model.wv[tok])
    if len(vectors) == 0:
        return np.zeros(w2v_model.wv.vector_size, dtype=float)
    return np.mean(vectors, axis=0)

if btn:
    if texto.strip() == "":
        st.warning("Se requiere un texto no vacío para realizar la predicción.")
    else:
        if artifacts["mode"] == "tfidf":
            pipe = artifacts["pipe"]
            pred = pipe.predict([texto])[0]
            st.success(f"ODS predicho: {int(pred)}")

            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba([texto])[0]
                top3 = proba.argsort()[::-1][:3]
                st.write("Top 3 ODS sugeridos (probabilidad):")
                for i in top3:
                    st.write(f"ODS {int(i)} → {float(proba[i]):.4f}")

        else:
            clf = artifacts["clf"]
            w2v = artifacts["w2v"]
            tokens = tokenize_basic(texto)
            vec = text_to_vec(tokens, w2v).reshape(1, -1)

            pred = clf.predict(vec)[0]
            st.success(f"ODS predicho: {int(pred)}")

            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(vec)[0]
                top3 = proba.argsort()[::-1][:3]
                st.write("Top 3 ODS sugeridos (probabilidad):")
                for i in top3:
                    st.write(f"ODS {int(i)} → {float(proba[i]):.4f}")
