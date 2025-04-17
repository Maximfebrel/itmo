import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pymatgen.core import Structure


class Loader:
    def __init__(self):
        self.train = None
        self.test = None
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.A_train = []
        self.A_test = []

        with open("config.yaml") as file:
            config = yaml.safe_load(file)

        self.load(config)

        self.make_graph()
        self.make_arrays()

    def make_arrays(self):
        self.X_train = np.array(self.X_train)
        self.X_test = np.array(self.X_test)
        self.y_train = np.array(self.y_train)
        self.y_test = np.array(self.y_test)
        self.A_train = np.array(self.A_train)
        self.A_test = np.array(self.A_test)

    def make_graph(self):
        max_len = []
        for idx, row in self.train.iterrows():
            max_len.append(len(row['structures'].sites))
        for idx, row in self.test.iterrows():
            max_len.append(len(row['structures'].sites))

        max_len = max(max_len)

        for idx, row in self.train.iterrows():
            self.X_train.append(self.make_features(row['structures'], max_len))
            scaler = StandardScaler()
            adj = row['structures'].distance_matrix
            adj_matrix = adj.reshape(adj.shape[0] ** 2, 1)
            scaler.fit(adj_matrix)
            adj_matrix = scaler.transform(adj_matrix)
            adj_matrix = adj_matrix.reshape(adj.shape[0], adj.shape[1])
            if len(row['structures'].distance_matrix) < max_len:
                r = max_len - len(row['structures'].distance_matrix)
                adj_matrix = np.pad(adj_matrix, (0, r), 'constant', constant_values=(4, 0))
            self.A_train.append(adj_matrix)
            self.y_train.append(row['targets'])

        for idx, row in self.test.iterrows():
            self.X_test.append(self.make_features(row['structures'], max_len))
            scaler = StandardScaler()
            adj = row['structures'].distance_matrix
            adj[adj < 0.5] = 0
            adj_matrix = adj.reshape(adj.shape[0] ** 2, 1)
            scaler.fit(adj_matrix)
            adj_matrix = scaler.transform(adj_matrix)
            adj_matrix = adj_matrix.reshape(adj.shape[0], adj.shape[1])
            if len(row['structures'].distance_matrix) < max_len:
                r = max_len - len(row['structures'].distance_matrix)
                adj_matrix = np.pad(adj_matrix, (0, r), 'constant', constant_values=(4, 0))
            self.A_test.append(adj_matrix)
            self.y_test.append(row['targets'])

    @staticmethod
    def make_features(data, max_len):
        # scaler = StandardScaler()
        graph = []

        # periodic = 1 if data.lattice.is_3d_periodic else 0
        # orthogonal = 1 if data.lattice.is_orthogonal else 0
        # ordered = 1 if data.is_ordered else 0
        #
        # categorial = np.array([periodic, orthogonal, ordered])
        # categorial = np.repeat(categorial[None, :], len(data.sites)).reshape(3, len(data.sites)).T
        #
        # numeric = np.array([float(data.density), data.volume, *data.lattice.parameters])
        # numeric = np.repeat(numeric[None, :], len(data.sites)).reshape(8, len(data.sites)).T

        names = []
        # Добавление узлов в граф
        for site in data:
            names.append(site.species.total_electrons)
            graph.append([*site.coords, *site.frac_coords, site.species.average_electroneg])

        graph = np.array(graph)
        # graph = np.hstack([graph, np.array(numeric)])
        # scaler.fit(graph)
        # graph = scaler.transform(graph)

        # graph = np.hstack([graph, categorial, np.array(names)[:, None]])
        graph = np.hstack([graph, np.array(names)[:, None]])
        zeros = np.zeros((max_len - len(graph), 8))
        graph = np.vstack([graph, zeros])

        return graph

    def load(self, config):
        self.train, self.test = self.prepare_dataset(config["datapath"])

    def prepare_dataset(self, dataset_path):
        dataset_path = Path(dataset_path)
        targets = pd.read_csv(dataset_path / "targets.csv", index_col=0)
        struct = {
            item.name.strip(".json"): self.read_pymatgen_dict(item)
            for item in (dataset_path / "structures").iterdir()
        }

        data = pd.DataFrame(columns=["structures"], index=struct.keys())
        data = data.assign(structures=struct.values(), targets=targets)

        scaler = StandardScaler()
        scaler.fit(data[['targets']])
        data['targets'] = scaler.transform(data[['targets']])

        return train_test_split(data, test_size=0.25, random_state=666)

    @staticmethod
    def read_pymatgen_dict(file):
        with open(file, "r") as f:
            d = json.load(f)
        return Structure.from_dict(d)
