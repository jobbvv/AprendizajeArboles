# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Cargar los datos desde un archivo Excel
df = pd.read_excel('dataset2.xlsx')

# Dividir los datos en características (X) y etiquetas (y)
X = df.drop('eficiente', axis=1)
y = df['eficiente']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear el modelo de árbol de decisiones
model = DecisionTreeClassifier()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Visualizar el árbol de decisión
plt.figure(figsize=(20,10))  # Tamaño de la figura
plot_tree(model, feature_names=X.columns, class_names=['No Eficiente', 'Eficiente'], filled=True, fontsize=10)  # Ajustar el tamaño de la fuente
plt.show()
