import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

n_cluster = 2
train_df = pd.read_csv('./processed.csv')
train_df.iloc[:, 3:train_df.shape[1]]
scaled_values = MinMaxScaler().fit_transform(train_df.iloc[:, 3:train_df.shape[1]])
cluster_indexs = KMeans(n_clusters = n_cluster).fit_predict(scaled_values)
print('--------------', scaled_values)

train_df['cluster_index'] = cluster_indexs
train_df.insert(0, 'no', [i + 1 for i in range(train_df.shape[0])])
print(train_df.head())
train_df.to_csv("output/flow_with_cluster_index.csv", index = False)

def plot(plt, gs, i, data_frame, column_name, plot_name, n_cluster, interval = 0.1):
    plt.subplot(gs[i, :])
    plt.title(plot_name or column_name) 
    for i in range(n_cluster):
        df_in_cluster = data_frame.ix[data_frame['cluster_index'] == i]
        x = np.array([(no - 1) * interval for no in df_in_cluster['no']])
        y = df_in_cluster[column_name].values
        plt.scatter(x, y)

plt.figure()
gs = gridspec.GridSpec(4, 1)
plot(plt, gs, 0, train_df, 'src_ent', 'src ip entropy', n_cluster)
plot(plt, gs, 1, train_df, 'dst_ent', 'dest ip entropy', n_cluster)
plot(plt, gs, 2, train_df, 'sport_ent', 'src port entropy', n_cluster)
plot(plt, gs, 3, train_df, 'dport_ent', 'dest port entropy', n_cluster)
plt.show()

