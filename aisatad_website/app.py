import streamlit as st
import datetime
import requests
import pandas as pd
import io

'''
# aisatad front
'''


st.title("Prédiction via FastAPI")

uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Aperçu du CSV :", df.head())

    if st.button("Envoyer pour prédiction"):
        # convertir le fichier Streamlit en binaire
        file_bytes = uploaded_file.getvalue()

        # envoyer le fichier avec le bon format attendu par FastAPI
        files = {
            "file": (uploaded_file.name, io.BytesIO(file_bytes), "text/csv")
        }

        response = requests.post("http://localhost:8000/predict", files=files)




        # stocker la réponse JSON
        if response.status_code == 200:
            data = response.json()
            st.write("Réponse brute de l’API :", data)

            if "anomaly" in data:
                predictions = data["anomaly"]
                st.success("Prédictions :")
                st.write(predictions)
            else:
                st.error("La réponse ne contient pas de prédictions.")
        else:
            st.error(f"Erreur lors de l'appel à l'API : {response.status_code}")
            st.text(response.text)
