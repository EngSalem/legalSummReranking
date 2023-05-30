import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('darkgrid')

df_valid = pd.read_csv('../../data/sum_pair/train_raw.csv')

df_valid['summary_length'] = df_valid.apply(lambda row: len(row['summary'].split()), axis=1)
df_valid = df_valid[df_valid['summary_length']<=512]

## clustering algorithm for vector quanitization
kmeans = KMeans(n_clusters=3, random_state=0)

## cluster the summary length
kmeans.fit(df_valid['summary_length'].values.reshape(-1,1))
df_valid['length_cluster'] = kmeans.labels_

## plot histogram

sns.histplot(data=df_valid, x='summary_length', hue='length_cluster')
plt.show()