import pandas as pd
from RNN import RNN
from LSTM import LSTM
from GRU import GRU
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
import datetime

data = pd.read_csv(r"C:\Users\makso\Desktop\ФООСИИ\recurent\Steel_industry_data.csv")

# производим кодировку категориальных переменных
le = preprocessing.OrdinalEncoder()
for col in ['WeekStatus', 'Day_of_week', 'Load_Type']:
    le.fit(np.array([data[col]]).T)
    data[col] = le.transform(np.array([data[col]]).T)

data = data.loc[data['Leading_Current_Reactive_Power_kVarh'] < 15]

columns_to_scale = ['Lagging_Current_Reactive.Power_kVarh',
                    'Leading_Current_Reactive_Power_kVarh',
                    'CO2(tCO2)', 'Lagging_Current_Power_Factor',
                    'Leading_Current_Power_Factor', 'NSM', 'Usage_kWh']

scaler = StandardScaler()
scaler.fit(data[columns_to_scale])
data[columns_to_scale] = scaler.transform(data[columns_to_scale])

features = data.drop(['Usage_kWh', 'date'], axis=1)
target = data['Usage_kWh'].copy()
dates = data['date'].copy()

data['date'] = data['date'].apply(lambda x: datetime.datetime.strptime(str(x), '%d/%m/%Y %H:%M'))
data.set_index('date', inplace=True)

X_train, X_test = features[:24510], features[24510:30030]
y_train, y_test = target[:24510], target[24510:30030]

model_rnn = RNN(9, 1)

loss_rnn = model_rnn.train(X=X_train, y=y_train, lr=1e-3, epochs=10, batch_size=10)

y_pred = model_rnn.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RNN MSE: {mse}')
print(f'RNN RMSE: {rmse}')
print(f'RNN R2: {r2}')
print(f"First epoch loss RNN - {loss_rnn[0]}\nLast epoch loss RNN - {loss_rnn[-1]}")
print()

model_lstm = LSTM(9, 1)

loss_lstm = model_lstm.train(X=X_train, y=y_train, lr=1e-2, epochs=9, batch_size=10)

y_pred = model_lstm.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'LSTM MSE: {mse}')
print(f'LSTM RMSE: {rmse}')
print(f'LSTM R2: {r2}')
print(f"First epoch loss LSTM - {loss_lstm[0]}\nLast epoch loss LSTM - {loss_lstm[-1]}")
print()

model_gru = GRU(9, 1)

loss_gru = model_gru.train(X=X_train, y=y_train, lr=1e-3, epochs=4, batch_size=10)

y_pred = model_gru.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'GRU MSE: {mse}')
print(f'GRU RMSE: {rmse}')
print(f'GRU R2: {r2}')
print(f"First epoch loss GRU - {loss_gru[0]}\nLast epoch loss GRU - {loss_gru[-1]}")
print()

plt.plot(loss_rnn)
plt.title('RNN')
plt.show()

plt.plot(loss_lstm)
plt.title('LSTM')
plt.show()

plt.plot(loss_gru)
plt.title('GRU')
plt.show()


