import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

data = pd.read_csv(r"C:\Users\makso\Desktop\АМО\KNN\iris.csv")

features = data.drop(['species'], axis=1)
target = data['species']

features_train, features_test, target_train, target_test = (
    train_test_split(features, target, test_size=0.25, random_state=24))


class KNN:
    def __init__(self, k=1):
        self.y_train = None
        self.X_train = None
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)

    def euclidean_distances(self, x_test_i):
        return np.sqrt(np.sum((self.X_train - x_test_i) ** 2, axis=1))

    @staticmethod
    def kernel(r):
        return (2 * np.pi) ** (-0.5) * np.exp(-0.5 * r ** 2)

    def _make_prediction(self, x_test_i):
        distances = self.euclidean_distances(x_test_i)

        weights = {}

        for _ in range(len(self.X_train)):
            cl = self.y_train[_]
            index_k_plus_1 = np.argsort(distances)[:self.k+1][-1]

            r = distances[_] / (distances[index_k_plus_1] + 10**(-4))

            try:
                weights[cl] += self.kernel(r)
            except KeyError:
                weights[cl] = self.kernel(r)

        max_ = 0
        max_key = 0
        for key, value in weights.items():
            if value > max_:
                max_key = key
                max_ = value

        return max_key

    def predict(self, X_test):
        return pd.Series([self._make_prediction(x) for x in np.array(X_test)])


class LOO:
    def __init__(self, model_, k):
        self.best_param_ = None
        self.model = model_
        self.k = k

    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        sum_s = {}

        for k_ in range(self.k):
            self.model.k = k_
            sum_ = 0

            for _ in range(len(X_train)):
                x_array = X_train[_]
                x_test = pd.DataFrame({'0': [x_array[0]], '1': [x_array[1]], '2': [x_array[2]], '3': [x_array[3]]})

                y_test = y_train[_]

                X_train_ = np.concatenate([X_train[0:_], X_train[_+1:]])
                y_train_ = np.concatenate([y_train[0:_], y_train[_+1:]])

                self.model.fit(X_train_, y_train_)
                pred = self.model.predict(x_test)

                if pd.Series(y_test)[0] != pred[0]:
                    sum_ += 1

            sum_s[k_] = sum_ / len(X_train)

        min_ = 10**9
        min_key = 0
        for key, value in sum_s.items():
            if value <= min_:
                min_key = key
                min_ = value

        self.best_param_ = min_key


search_params = LOO(model_=KNN(), k=10)
search_params.fit(features_train, target_train)

model = KNN(search_params.best_param_)
model.fit(features_train, target_train)
target_pred = model.predict(features_test)

# подбор гиперпараметров
clf = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': range(1, 10)})
clf.fit(features_train, target_train)

best_model = KNeighborsClassifier(**clf.best_params_)
best_model.fit(features_train, target_train)
predicted = best_model.predict(features_test)

print(accuracy_score(target_pred, target_test), accuracy_score(predicted, target_test))
