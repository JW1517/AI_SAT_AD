import streamlit as st
import datetime
import requests
import pandas as pd
import io
import ast
import matplotlib.pyplot as plt


'''
# aisatad front
'''

def plot_seg_ax_st(ax, df_raw, segment):
    df_filtered = df_raw[df_raw["segment"] == segment]
    ax.scatter(df_filtered["timestamp"], df_filtered["value"])
    ax.set_title(f"Segment {segment}\n | Channel {df_filtered['channel'].iloc[0]}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Value")


st.title("Prédiction via FastAPI")

uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])
base_url = "https://aisatad-img-204615645613.europe-west1.run.app/"


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Aperçu du CSV :", df.head())

    nb_seg_plot = 5
    fig, ax = plt.subplots(nb_seg_plot, figsize=(12, 17))
    segments = df["segment"].unique()[:nb_seg_plot]
    for i, segment in enumerate(segments):
        plot_seg_ax_st(ax[i], df, segment)

    plt.tight_layout()
    st.pyplot(fig)

    if st.button("Envoyer pour prédiction"):
        url_post= base_url + "/predict"
        res = ast.literal_eval(df.to_json())
        response = requests.post(url_post, json = res)

        if response.status_code == 200:
            data = response.json()
            st.write("Réponse brute de l’API :", data)

        else:
            st.error(f"Erreur lors de l'appel à l'API : {response.status_code}")
            st.text(response.text)
