import matplotlib.pyplot as plt

def plot_seg(df_raw, list_seg):
    for segment in list_seg:
        df_filtered = df_raw[df_raw["segment"] == segment]
        plt.figure(figsize=(15, 5))
        plt.scatter(df_filtered["timestamp"], df_filtered["value"])
        plt.title(f"Segment {segment} - Anomaly {df_filtered['anomaly'].iloc[0]} - Channel {df_filtered['channel'].iloc[0]} - Sampling {df_filtered['sampling'].iloc[0]}")
        plt.tight_layout()
        print(plt.show())
