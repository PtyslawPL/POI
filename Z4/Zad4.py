import numpy as np
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Input
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Wczytanie danych z pliku CSV
features = pd.read_csv("vectors_100.csv", sep=',') # Otworzenie csv
data = np.array(features) # Konwersja na tablicę
X = (data[:,:-1]).astype('float64') # Wyodrębnianie wektora cech do macierzy X
Y = data[:,-1] # Wyodrębnianie etykiety kategorii do wektora Y

# Kodowanie etykiet klas jako liczby całkowite
label_encoder = LabelEncoder() # Inicjalizacja przekształtnika etykiet na liczby całkowite
y_int = label_encoder.fit_transform(Y) # Przekształcenie etykiet na liczby całkowite

# Kodowanie etykiet klas One-hot
onehot_encoder = OneHotEncoder(sparse_output=False) # Inicjalizacja kodera One-hot
y_onehot = onehot_encoder.fit_transform(y_int.reshape(-1, 1)) # Przekształcenie etykiet na kodowanie One-hot

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3) # Podział danych na 70% treningowych i 30% testowych

# Tworzenie modelu sieci neuronowej
n_classes = y_train.shape[1] # Liczba klas na podstawie liczby kolumn w y_train
model = Sequential() # Inicjalizacja modelu sekwencyjnego
input_dim = X.shape[1]  # Dynamicznie wyznanaczanie liczba kolumn w zbiorze cech
model.add(Input(shape=(input_dim,))) # Dodanie warstwy wejściowej
model.add(Dense(10, activation='sigmoid')) # Dodanie warstwy ukrytej z 10 neuronami i funkcją aktywacji sigmoid
model.add(Dense(n_classes, activation='softmax')) # Dodanie warstwy wyjściowej z n_classes neuronami i funkcją aktywacji softmax

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) # Kompilacja modelu z funkcją straty 'categorical_crossentropy', optymalizatorem 'sgd' i metryką 'accuracy'

# Uczenie sieci
model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True) # Trenowanie modelu przez 100 epok z rozmiarem partii 10 i losowym przetasowaniem danych

# Testowanie sieci
y_pred = model.predict(X_test) # Predykcja na zbiorze testowym
y_pred_classes = np.argmax(y_pred, axis=1) # Wyodrębnienie klas z predykcji
y_test_classes = np.argmax(y_test, axis=1) # Wyodrębnienie klas z etykiet testowych

conf_matrix = confusion_matrix(y_test_classes, y_pred_classes) # Obliczenie macierzy pomyłek
print(conf_matrix) # Wyświetlenie macierzy pomyłek