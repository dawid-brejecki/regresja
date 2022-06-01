
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor



df=pd.read_csv('output.csv')

# wybranie tylko interesujacych kolumn ze zbioru
df = df.iloc[:,[4,6,7,8,9,10,11]]

# zmiana nazw kolumn dla czytelnosci
nowe_nazwy = ['happiness','economy','family','health','freedom','trust','social']
df.columns = nowe_nazwy

# z uwagi na dosc duza liczbe danych, w tym przypadku, usuwam wiersze z brakami, zamiast np. zastapienia ich srednia
df = df.dropna()

# rzut oka na dane
sns.set_theme(style="ticks")
sns.pairplot(df)


# macierz korelacji
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)



# Z uwagi na niska korelacje ze zmienna objasniana, nie uwzgledniam w modelu zmiennych trust i social.
# Ponadto, z uwagi na wysoka korelacje zmiennej economy i health, odrzucam zmienna health.
# Wykresy wskazuja na zasadnosc uzycia regresji liniowej

# przygotowanie X i Y
Y = df['happiness']
features = ['economy','family','freedom']
X = df[features]

# podzial zbioru na testowy i treningowy
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)

# sprawdzenie wynikow zbiory testowego za pomoca walidacji krzyzowej
classifier = LinearRegression()
scores = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=10)
print(f'Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})')
scores = pd.DataFrame(scores, columns=['accuracy'])
px.bar(scores, y='accuracy', color='accuracy', width=700, height=400, 
       title=f'Walidacja krzyżowa | Accuracy: {scores.mean()[0]:.4f} (+/- {scores.std()[0]:.3f})',
       color_continuous_scale=px.colors.sequential.Inferno_r, range_color=[scores.min()[0] - 0.01, 1.0])


# regresja
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# predykcja
Y_pred = regressor.predict(X_test)

# sredni blad
np.sqrt(mean_squared_error(Y_test, Y_pred))

# wspolczynnik determinancji
regressor.score(X_test, Y_test)

# wspolczynnik modelu
regressor.coef_




regressor.score(X_test, Y_test)





# histogram bledu predykcji
predictions = pd.DataFrame(data={'y_true': Y_test, 'y_pred': Y_pred})
predictions['error'] = predictions['y_true'] - predictions['y_pred']
_ = predictions['error'].plot(kind='hist', bins=50, figsize=(7, 7))





# wykres modelu

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('wartosc rzeczywista')
plt.ylabel('wartosc przewidziana')
plt.scatter(Y_test, Y_pred)
plt.plot(Y_pred, Y_pred, color='red', label='model')
plt.legend()
plt.show()





# regresja z wykorzystaniem metody stochastycznego spadku gradientu

regressor = SGDRegressor()
regressor.fit(X_train, Y_train)

# predykcja
Y_pred = regressor.predict(X_test)

# sredni blad
np.sqrt(mean_squared_error(Y_test, Y_pred))

# wspolczynnik determinancji
regressor.score(X_test, Y_test)





# regresja z wykorzystaniem maszyny wektorow nosnych

regressor = SVR(kernel='linear')
regressor.fit(X_train, Y_train)

# predykcja
Y_pred = regressor.predict(X_test)

# sredni blad
np.sqrt(mean_squared_error(Y_test, Y_pred))

# wspolczynnik determinancji
regressor.score(X_test, Y_test)





regressor = SVR(kernel='rbf')
regressor.fit(X_train, Y_train)

# predykcja
Y_pred = regressor.predict(X_test)

# sredni blad
np.sqrt(mean_squared_error(Y_test, Y_pred))

# wspolczynnik determinancji
regressor.score(X_test, Y_test)




# regresja z XGBoost

# wyszukiwanie optymalnych parametrow
xgb1 = XGBRegressor()
parameters = {'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], 
              'max_depth': [6,12,18],
              'min_child_weight': [6, 12, 18],
              'subsample': [0.5, 0.7, 1],
              'colsample_bytree': [0.7, 0.9],
              'n_estimators': [100, 500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=0)

xgb_grid.fit(X_train, Y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)





# aplikacja najlepszych parametrow do modelu
regressor = XGBRegressor(**xgb_grid.best_params_)
regressor.fit(X_train, Y_train)

# predykcja
Y_pred = regressor.predict(X_test)

# sredni blad
np.sqrt(mean_squared_error(Y_test, Y_pred))

# wspolczynnik determinancji
regressor.score(X_test, Y_test)





# sprawdzenie skutecznosci innych modeli
lista = []
regressors = [BayesianRidge(),
              DecisionTreeRegressor(),
              ]
for reg in regressors:
    regressor = reg
    regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    wynik = regressor.score(X_test, Y_test)
    lista.append(wynik)
    
print(lista)




# Najlepszym estymatorem okazał się SGD Regressor, ale z uwagi na charakter danych 
# wiekszosc innych estymatorow także okazało się być skutecznych

