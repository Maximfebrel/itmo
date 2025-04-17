import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import Loader
from arch import CNN

data = Loader()

# разделение выборки на тренировочную и тестовую
features = data.x_train
target = data.y_train
X_val, features_train, y_val, target_train = train_test_split(features, target, test_size=0.06, random_state=42)

X_, features_test, y_, target_test = train_test_split(X_val, y_val, test_size=0.02, random_state=42)

# инициализация модели
model = CNN()
# обучение модели
model.train(features_train, target_train, 10, 1e-4, 3000, 250)
# предсказание
y_pred, thresh = model.predict(features_test)

y_pred = pd.Series(y_pred)
y_pred.to_excel(r"C:\Users\makso\Desktop\ФООСИИ\CNN\data\pred.xlsx", index=False)

thresh.to_excel(r"C:\Users\makso\Desktop\ФООСИИ\CNN\data\thresh.xlsx", index=False)

targ = []
for test in target_test:
    targ.append(float(test))

targ = pd.Series(targ)
targ.to_excel(r"C:\Users\makso\Desktop\ФООСИИ\CNN\data\targ.xlsx", index=False)
