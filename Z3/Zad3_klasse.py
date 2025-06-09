import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

dataset = input("Podaj ścieżkę do pliku csv z kategoriami tekstur: ").strip()
features = pd.read_csv(dataset, sep=',')

data = np.array(features)
X = (data[:,:-1]).astype('float64')
Y = data[:,-1] # etykiety klas

x_transform = PCA(n_components=3)

Xt = x_transform.fit_transform(X)

red = Y == 'Aluminium'
green = Y == 'Ceramika'
yellow = Y == 'Tekstylia'
blue = Y == 'Tynk'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xt[red, 0], Xt[red, 1], Xt[red, 2], c="r", label="Aluminium")
ax.scatter(Xt[green, 0], Xt[green, 1], Xt[green, 2], c="g", label="Ceramika")
ax.scatter(Xt[yellow, 0], Xt[yellow, 1], Xt[yellow, 2], c="y", label="Tekstylia")
ax.scatter(Xt[blue, 0], Xt[blue, 1], Xt[blue, 2], c="b", label="Tynk")
ax.legend(loc='upper right', fontsize='small')

classifier = svm.SVC(gamma='auto') # klasyfikator SVM

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33) 

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print("Dokładność klasyfikatora wynosi: " + str(acc) + "\n")

cm = confusion_matrix(y_test, y_pred, normalize='true')

print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Aluminium','Ceramika','Tekstylia','Tynk'])
disp.plot(cmap='Blues')
plt.show()