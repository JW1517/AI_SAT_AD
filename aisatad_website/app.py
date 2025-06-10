import streamlit as st
import datetime
import requests
import pandas as pd
import io
import ast
from aisatad.function_files.plot import plot_seg_ax,plot_seg_ax_st
import matplotlib.pyplot as plt


'''
# aisatad front
'''


st.title("Prédiction via FastAPI")

uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])
base_url = "http://localhost:8000"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)#,parse_dates=["timestamp"])
    st.write("Aperçu du CSV :", df.head())

    # list_seg =df["segment"].unique().tolist()
    # fig,ax = plt.subplots()

    # for segment in list_seg[:2]:
    #     df_filtered = df[df["segment"] == segment]
    #     plt.figure(figsize=(15, 5))
    #     ax.scatter(df_filtered["timestamp"], df_filtered["value"])
    #     plt.title(f"Segment {segment} - Anomaly {df_filtered['anomaly'].iloc[0]} - Channel {df_filtered['channel'].iloc[0]} - Sampling {df_filtered['sampling'].iloc[0]}")
    #     plt.tight_layout()
    #     st.pyplot(fig)

    nb_seg_plot = 5  # choisi combient de segment tu veut plotter pour chaque capteur

    fig, ax = plt.subplots(nb_seg_plot, figsize=(12, 17))


    segments = df["segment"].unique()[:nb_seg_plot]


    for i, segment in enumerate(segments):
        plot_seg_ax_st(ax[i], df, segment)
        # min_data = df[df["segment"]==segment]["timestamp"].min()
        # max_data = df[df["segment"]==segment]["timestamp"].max()
        # ax[i].set_xlim(min_data, max_data)
        # mid_data = max_data - (max_data - min_data)/2
        # ax[i].set_xticks([min_data, mid_data, max_data])







    # plt.locator_params(axis='x',bins =5)
    plt.tight_layout()
    st.pyplot(fig)




    if st.button("Envoyer pour prédiction"):
        # convertir le fichier Streamlit en binaire
        url_post= base_url + "/predict"
        res = ast.literal_eval(df.to_json())
        #res = df.head(10).to_json()
        response = requests.post(url_post, json = res)





        # stocker la réponse JSON
        if response.status_code == 200:
            data = response.json()
            st.write("Réponse brute de l’API :", data)






            #st.json(data)

            # if "anomaly" in data:
            #     predictions = data["anomaly"]
            #     st.success("Prédictions :")
            #     st.write(predictions)
            # else:
            #     st.error("La réponse ne contient pas de prédictions.")
        else:
            st.error(f"Erreur lors de l'appel à l'API : {response.status_code}")
            st.text(response.text)
