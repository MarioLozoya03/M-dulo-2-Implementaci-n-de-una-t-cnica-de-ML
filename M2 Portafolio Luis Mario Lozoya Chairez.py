import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt

# Carga el archivo CSV en un DataFrame de pandas
data = pd.read_csv('Admission_Predict.csv')

# Muestra las primeras filas del DataFrame para verificar que se leyó correctamente
# print(data.head())

# Preprocesamiento inicial de datos
# Seleccionar las columnas relevantes para X (características) y y (etiqueta objetivo)
X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']].values
y = data['Chance of Admit '].values

# Normalización de los datos para mejorar la convergencia durante el entrenamiento
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Definición de las funciones necesarias para el algoritmo de regresión lineal

# Función para actualizar los pesos (w) y el sesgo (b) usando el descenso de gradiente
def update_w_and_b(X, y, w, b, alpha):
    dl_dw = np.zeros_like(w)  # Gradiente de la pérdida respecto a w
    dl_db = 0.0  # Gradiente de la pérdida respecto a b
    N = len(X)  # Número de ejemplos en el dataset
    for i in range(N):
        y_pred = np.dot(X[i], w) + b  # Predicción del modelo
        error = y[i] - y_pred  # Error de predicción
        dl_dw += -2 * X[i] * error  # Actualización del gradiente de w
        dl_db += -2 * error  # Actualización del gradiente de b
    # Actualizar w y b usando la tasa de aprendizaje alpha
    w = w - (1 / N) * dl_dw * alpha
    b = b - (1 / N) * dl_db * alpha
    return w, b

# Función para calcular la pérdida media cuadrática
def avg_loss(X, y, w, b):
    N = len(X)  # Número de ejemplos en el dataset
    total_error = 0.0
    for i in range(N):
        y_pred = np.dot(X[i], w) + b  # Predicción del modelo
        total_error += (y[i] - y_pred) ** 2  # Suma de los errores cuadrados
    return total_error / float(N)  # Promedio de los errores cuadrados

# Función para entrenar el modelo y graficar la pérdida durante el entrenamiento
def train_and_plot(X, y, w, b, alpha, epochs):
    loss_history = []  # Historial de la pérdida para graficar
    for e in range(epochs):
        w, b = update_w_and_b(X, y, w, b, alpha)  # Actualizar los pesos y el sesgo
        loss = avg_loss(X, y, w, b)  # Calcular la pérdida actual
        loss_history.append(loss)  # Guardar la pérdida en el historial
        if e % 100 == 0:
            print(f"Epoch {e} | Pérdida: {loss:.4f} | w:{w}, b:{b:.4f}")  # Mostrar progreso cada 100 epochs
    # Graficar la pérdida a lo largo de los epochs
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), loss_history, label='Pérdida')
    plt.xlabel('Epochs')
    plt.ylabel('Pérdida')
    plt.title('Progreso del Entrenamiento de Regresión Lineal')
    plt.legend()
    plt.grid(True)
    plt.show()
    return w, b

# Función para hacer predicciones con el modelo entrenado
def predict(x, w, b):
    return np.dot(x, w) + b

# Clase para la interfaz gráfica de usuario (GUI) usando Tkinter
class LinearRegressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Regresión Lineal")
        self.root.geometry("500x300")
        self.w = None  # Inicialización de los pesos
        self.b = None  # Inicialización del sesgo
        self.alpha = 0.01  # Tasa de aprendizaje por defecto
        self.epochs = 1000  # Número de epochs por defecto

        # Configuración de los elementos gráficos (labels, buttons, etc.)
        tk.Label(root, text="Regresión Lineal - Predicción de Admisión", font=("Arial", 14)).pack(pady=10)
        tk.Label(root, text="Tasa de aprendizaje (alpha):").pack(pady=5)
        
        self.alpha_entry = tk.Entry(root)
        self.alpha_entry.insert(0, "0.01")  # Valor inicial de alpha
        self.alpha_entry.pack(pady=5)
        
        tk.Label(root, text="Número de Epochs:").pack(pady=5)
        self.epochs_entry = tk.Entry(root)
        self.epochs_entry.insert(0, "1000")  # Valor inicial de epochs
        self.epochs_entry.pack(pady=5)

        # Botones para cargar el dataset, entrenar el modelo y hacer predicciones
        tk.Button(root, text="Cargar Dataset", command=self.load_data).pack(pady=5)
        tk.Button(root, text="Entrenar Modelo", command=self.train_model).pack(pady=5)
        tk.Button(root, text="Predecir", command=self.predict).pack(pady=5)

    # Función para cargar el dataset desde un archivo CSV
    def load_data(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.data = pd.read_csv(filepath)
            self.prepare_data()
            messagebox.showinfo("Información", "Dataset cargado exitosamente.")

    # Función para preprocesar los datos cargados
    def prepare_data(self):
        self.X = self.data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']].values
        self.y = self.data['Chance of Admit '].values
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)  # Normalización de las características

    # Función para entrenar el modelo con los datos preprocesados
    def train_model(self):
        if hasattr(self, 'X') and hasattr(self, 'y'):
            self.alpha = float(self.alpha_entry.get())  # Obtener el valor de alpha de la entrada
            self.epochs = int(self.epochs_entry.get())  # Obtener el número de epochs de la entrada
            self.w, self.b = np.zeros(self.X.shape[1]), 0.0  # Inicialización de w y b
            self.w, self.b = train_and_plot(self.X, self.y, self.w, self.b, self.alpha, self.epochs)
            messagebox.showinfo("Información", "Modelo entrenado exitosamente.")
        else:
            messagebox.showwarning("Advertencia", "Primero cargue el dataset.")

    # Función para predecir un valor utilizando el modelo entrenado
    def predict(self):
        if self.w is not None and self.b is not None:
            x_new = self.X[0]  # Selecciona un ejemplo para predecir
            y_pred = predict(x_new, self.w, self.b)
            messagebox.showinfo("Predicción", f"Predicción: {y_pred:.4f}")
        else:
            messagebox.showwarning("Advertencia", "Primero entrene el modelo.")

# Crear la ventana principal y ejecutar la GUI
root = tk.Tk()
app = LinearRegressionGUI(root)
root.mainloop()
