from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data_iris = pd.read_csv("iris.csv")
data_iris = data_iris.drop_duplicates()

X_iris = data_iris.drop(['species'], axis=1)
y_iris = data_iris['species']

cluster_df_iris = []

for idx_2, row_2 in X_iris.iterrows():
    cluster_df_iris.append(tuple(row_2))


# иерархическая кластеризация
def hier(cluster_, clast_num):
    clusters = [cluster_]
    distances = dict()
    list_indexes_1 = []
    list_indexes_2 = []
    list_dists = []
    list_lens = []

    for i in range(1, len(cluster_) + 1):
        min_dist = 10 ** 9

        if len(clusters[-1]) == 1:
            max_ = 0
            for _ in range(len(list_dists)):
                if _ != len(list_dists) - 1:
                    if abs(list_dists[_ + 1] - list_dists[_]) > max_:
                        max_ = abs(list_dists[_ + 1] - list_dists[_])
                        index_ = _

            matrix = np.array([list_indexes_1, list_indexes_2, list_dists, list_lens])
            return clusters[index_], matrix

        if i == 1:
            for k in range(len(clusters[i - 1])):
                for j in range(len(clusters[i - 1])):
                    if k != j:
                        dist = np.linalg.norm(np.array(clusters[i - 1][k]) - np.array(clusters[i - 1][j]))
                        distances[(clusters[i - 1][k], clusters[i - 1][j])] = dist

                        if dist < min_dist:
                            min_dist = dist
                            min_1 = clusters[i - 1][k]
                            min_2 = clusters[i - 1][j]
        else:
            for key, value in distances.items():
                if value < min_dist:
                    min_dist = value
                    min_1 = key[0]
                    min_2 = key[1]
        if i == 1:
            clust_for_indexes = clusters[-1]
        else:
            clust_for_indexes.append(new_cluster)

        new_cluster = (min_1, min_2)

        index_1 = clust_for_indexes.index(min_1)
        index_2 = clust_for_indexes.index(min_2)
        dist_ = min_dist
        len_1 = len(post_process(min_1))
        len_2 = len(post_process(min_2))
        total_len = len_1 + len_2

        list_indexes_1.append(index_1)
        list_indexes_2.append(index_2)
        list_dists.append(dist_)
        list_lens.append(total_len)

        cluster_copy = clusters[-1].copy()
        cluster_copy.remove(min_1)
        cluster_copy.remove(min_2)
        clusters.append([*cluster_copy])

        for s in clusters[-1]:
            v_s = distances.get((s, new_cluster[0]))
            if v_s is None:
                v_s = distances.get((new_cluster[0], s))

            u_s = distances.get((s, new_cluster[1]))
            if u_s is None:
                u_s = distances.get((new_cluster[1], s))

            distances[(new_cluster, s)] = 0.5 * (v_s + u_s + abs(v_s - u_s))

        keys_ = []
        for key_ in distances.keys():
            if min_2 in key_ or min_1 in key_:
                keys_.append(key_)

        for k_ in keys_:
            del distances[k_]

        clusters[-1].append(new_cluster)


# постобработка выхода иерархической кластеризации
def post_process(cl):
    i = 0
    list_ = []
    for symb in str(cl):
        if symb == '(' and str(cl)[i + 1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
            list_.append(str(cl)[i - 1: str(cl)[i - 1:].find(')') + i].replace('(', '').replace(')', ''))
        else:
            pass
        i += 1
    return list_


def create_str(list_1):
    str_to_df = []
    for i_ in list_1:
        str_to_df_ = []
        for j_ in i_.split(','):
            str_to_df_.append(float(j_))
        str_to_df.append(str_to_df_)
    return str_to_df


cl, matrix_ = hier(cluster_df_iris, 4)
