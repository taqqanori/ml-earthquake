import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm

def swarm_plot():
    fig, ax = plt.subplots(figsize=(8,6))
    countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    countries.plot(color='lightgrey', ax=ax)
    df = pd.read_csv('data/swarms.csv')
    df.plot(x='longitude', y='latitude', kind='scatter', s=0.01, alpha=0.1, color='red', ax=ax)
    plt.show()

if __name__ == '__main__':
    swarm_plot()
