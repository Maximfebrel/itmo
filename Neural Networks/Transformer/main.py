import pandas as pd

from Model.Transformer import Transformer
from loader import Loader

loader = Loader()

X_train, X_test, y_train, y_test = loader.X_train, loader.X_test, loader.y_train, loader.y_test

input_dim = X_train.shape[1]
embed_dim = 1024
num_heads = 32
ff_dim = 1024
num_layers = 2

transformer = Transformer(num_layers=num_layers, input_dim=input_dim, embed_dim=embed_dim, num_heads=num_heads,
                          ff_dim=ff_dim, butch_size=10, epochs=50, learning_rate=1e-4)

loss = transformer.train(X_train, y_train)
pred, out = transformer.predict(X_test)

pd.Series(pred).to_excel(r"C:\Users\makso\Desktop\ФООСИИ\transformer\Pred.xlsx", index=False)
pd.Series(out).to_excel(r"C:\Users\makso\Desktop\ФООСИИ\transformer\Out.xlsx", index=False)
y_test.to_excel(r"C:\Users\makso\Desktop\ФООСИИ\transformer\True.xlsx", index=False)
pd.Series(loss).to_excel(r"C:\Users\makso\Desktop\ФООСИИ\transformer\Loss.xlsx", index=False)


