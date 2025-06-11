import streamlit as st
import requests
import pandas as pd
import ast
import matplotlib.pyplot as plt


def plot_seg_ax_st(ax, df_raw, segment):
    df_filtered = df_raw[df_raw["segment"] == segment]
    ax.scatter(df_filtered["timestamp"], df_filtered["value"], s=1)
    ax.set_xticks([])
    ax.set_yticks([])



st.title("AI SAT AD")

uploaded_file = st.file_uploader("Uploader des mesures (format CSV)", type=["csv"])
base_url = "https://aisatad-img-204615645613.europe-west1.run.app/"


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    nb_seg_plot = 5
    fig, ax = plt.subplots(2,3, figsize=(6, 3))
    segments = df["segment"].unique()[:nb_seg_plot]

    for ax, segment in zip(ax.flat, segments) :
        plot_seg_ax_st(ax, df, segment)

    plt.tight_layout()
    st.pyplot(fig)

    if st.button("Détecter les anomalies"):
        url_post= base_url + "/predict"
        res = ast.literal_eval(df.to_json())
        response = requests.post(url_post, json = res)

        if response.status_code == 200:
            data = response.json()
            for key, value in data.items() :
                if value == 0 :
                    st.write("Segment numéro", key, 'OK')
                else :
                    st.write("Segment numéro", key, 'ANOMALIE')
        else:
            st.error(f"Erreur lors de l'appel à l'API : {response.status_code}")
            st.text(response.text)
