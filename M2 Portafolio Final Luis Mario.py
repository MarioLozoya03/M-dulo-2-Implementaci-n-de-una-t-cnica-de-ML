import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Carga el archivo CSV en un DataFrame de pandas
data = pd.read_csv('Admission_Predict.csv')

# Seleccionar las columnas relevantes para X (características) y y (etiqueta objetivo)
X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']].values
y = data['Chance of Admit '].values

# Separar los datos en conjunto de entrenamiento (80%) y conjunto de prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización de los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del modelo de regresión lineal usando Scikit-learn
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Hacer predicciones
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Evaluación del modelo: cálculo del MSE y R2
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"MSE Entrenamiento: {mse_train:.4f}")
print(f"MSE Prueba: {mse_test:.4f}")
print(f"R^2 Entrenamiento: {r2_train:.4f}")
print(f"R^2 Prueba: {r2_test:.4f}")

# Gráfica comparativa de los errores en entrenamiento y prueba
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Valores Reales')
plt.plot(y_test_pred, label='Predicciones', linestyle='--')
plt.xlabel('Índice')
plt.ylabel('Probabilidad de Admisión')
plt.title('Comparación de Predicciones y Valores Reales en Conjunto de Prueba')
plt.legend()
plt.grid(True)
plt.show()

# Aplicación de regularización Ridge y Lasso
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_test_pred_ridge = ridge_model.predict(X_test_scaled)

lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train_scaled, y_train)
y_test_pred_lasso = lasso_model.predict(X_test_scaled)

# Evaluar modelos regularizados
mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge)
mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)
r2_test_ridge = r2_score(y_test, y_test_pred_ridge)
r2_test_lasso = r2_score(y_test, y_test_pred_lasso)

print(f"MSE Ridge: {mse_test_ridge:.4f}")
print(f"MSE Lasso: {mse_test_lasso:.4f}")
print(f"R^2 Ridge: {r2_test_ridge:.4f}")
print(f"R^2 Lasso: {r2_test_lasso:.4f}")

# Gráfica comparativa del rendimiento de los modelos regularizados
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Valores Reales')
plt.plot(y_test_pred_ridge, label='Predicciones Ridge', linestyle='--')
plt.plot(y_test_pred_lasso, label='Predicciones Lasso', linestyle=':')
plt.xlabel('Índice')
plt.ylabel('Probabilidad de Admisión')
plt.title('Comparación de Predicciones: Ridge vs Lasso')
plt.legend()
plt.grid(True)
plt.show()