import pandas as pd
import numpy as np

from Loader import Loader
from GraphRegressor import GraphRegressor

from sklearn.metrics import r2_score, root_mean_squared_error as rmse, mean_squared_error as mse, mean_absolute_error as mae
import matplotlib.pyplot as plt

loader = Loader()

X_train, X_test, y_train, y_test, A_train, A_test = loader.X_train, loader.X_test, loader.y_train, loader.y_test, loader.A_train, loader.A_test

input_dim = 8
hidden_dim = 10
num_nodes = 192

gnn = GraphRegressor(input_dim, hidden_dim, num_nodes, 2e-4, 16, 10)

loss = gnn.fit(A_train, X_train, y_train)

plt.plot(loss)
plt.show()

pred = gnn.predict(A_test, X_test)
pred = pd.Series(pred.reshape(742))
true = pd.Series(y_test)

print('r2_score: ', r2_score(pred, true),
      'rmse: ', rmse(pred, true),
      'mse: ', mse(pred, true),
      'mae: ', mae(pred, true),
      sep='\n')


pd.Series(loss).to_excel(r"C:\Users\makso\Desktop\ФООСИИ\GNN\loss.xlsx")
pred.to_excel(r"C:\Users\makso\Desktop\ФООСИИ\GNN\pred.xlsx")
true.to_excel(r"C:\Users\makso\Desktop\ФООСИИ\GNN\true.xlsx")
