import streamlit as st
import datetime
import requests
import pandas as pd
import io
import ast
#from aisatad.function_files.plot import plot_seg_ax,plot_seg_ax_st
import matplotlib.pyplot as plt


'''
# aisatad front
'''
#streamlit sans étiquette "anomaly"
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
    df = pd.read_csv(uploaded_file)#,parse_dates=["timestamp"])
    st.write("Aperçu du CSV :", df.head())



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
