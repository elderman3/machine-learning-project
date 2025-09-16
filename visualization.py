import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json, pandas as pd, plotly.express as px
from sklearn.decomposition import PCA

df = pd.read_csv("s2_worldcover_samples.csv")

coords = df[".geo"].apply(lambda s: json.loads(s)["coordinates"])
df["lon"] = coords.apply(lambda c: c[0])
df["lat"] = coords.apply(lambda c: c[1])
features = ["B2","B3","B4","B8","B11","B12"]
df_small = df[features + ["label"]]
n = min(2000, len(df_small))
df_small = df_small.sample(n, random_state=0)

def overlay_map_of_area():
    palette = ['#006400','#bbd17a','#ffff00','#f096ff','#fa0000',
            '#b4b4b4','#f0f0f0','#0064ff','#00ffff','#00aaff','#ffaaff']
    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon", color="label",
        color_discrete_sequence=palette, zoom=10, height=700, opacity=0.6
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0,r=0,t=0,b=0))
    fig.write_html("samples_map.html")

def box_polts_of_distributions():
    
    sns.pairplot(df_small, vars=features, hue="label", plot_kws={"s":10, "alpha":0.6})
    plt.show()
    plt.figure(figsize=(15,8))
    for i, band in enumerate(features, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(data=df_small, x="label", y=band, showfliers=False)
        plt.title(band)
    plt.tight_layout()
    plt.show()


def simple_pca_on_components():
    X = df_small[features].values
    y = df_small["label"].values
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    plt.scatter(coords[:,0], coords[:,1], c=y, cmap="tab10", s=10, alpha=0.6)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA of features")
    plt.show()

overlay_map_of_area()
box_polts_of_distributions()
simple_pca_on_components()
