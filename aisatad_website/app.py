import streamlit as st
import requests
import pandas as pd
import ast
import altair as alt

st.title("AI SAT AD")

uploaded_file = st.file_uploader("Uploader des mesures (format CSV)", type=["csv"])

base_url = "https://aisatad-img-204615645613.europe-west1.run.app/"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    #st.write("Aperçu des données chargées :")
    #st.write(df.head())

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    timestamp_type = 'T'  # Temporal axis for Altair

    nb_seg_plot = 6
    segments = df["segment"].unique()[:nb_seg_plot]

    st.subheader("Mesures par segment")

    cols = st.columns(1)
    for i, segment in enumerate(segments):
        df_filtered = df[df["segment"] == segment]

        scatter_chart = alt.Chart(df_filtered).mark_circle(size=10).encode(
            x=alt.X('timestamp:' + timestamp_type, title='Timestamp'),
            y=alt.Y('value:Q', title='Valeur'),
            tooltip=['timestamp', 'value']
        ).properties(
            width=400,
            height=250,
            title=f"Segment n° {i+1}"
        ).interactive()

        with cols[i % 1]:
            st.altair_chart(scatter_chart, use_container_width=True)

#if st.button("Détecter les anomalies"):
    url_post = base_url + "/predict"

    df_to_send = df.copy()
    df_to_send['timestamp'] = df_to_send['timestamp'].astype(str)

    res = ast.literal_eval(df_to_send.to_json())
    response = requests.post(url_post, json=res)

    if response.status_code == 200:
        st.balloons()
        data = response.json()
        for i, (key, value) in enumerate(data.items()):
            if value == 0:
                st.success(f"Segment n° {i+1} Nominal")
            else:
                st.error(f"Segment n° {i+1} ANOMALIE")
    else:
        st.text(f"Erreur lors de l'appel à l'API : {response.status_code}")
        st.text(response.text)

#st.subheader("Performance du modèle")
#col1, col2, col3, col4 = st.columns(4)
#col1.metric("Accuracy", "98.3%", "+0.6%")
#col2.metric("Precision", "99%", "+2.7%")
#col3.metric("Recall", "94.7%", "+1.8%")
#col4.metric("F1", "96%", "+1.4%")
