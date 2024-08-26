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
# Seleccionar las columnas relevantes para X y y
X = data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']].values
y = data['Chance of Admit '].values

# Normalización de los datos para mejorar la convergencia durante el entrenamiento
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Definición de las funciones necesarias para el algoritmo de regresión líneal

# Definición de las funciones necesarias para el algoritmo de regresión lineal
def update_w_and_b(X, y, w, b, alpha):
    dl_dw = np.zeros_like(w)
    dl_db = 0.0
    N = len(X)
    for i in range(N):
        y_pred = np.dot(X[i], w) + b
        error = y[i] - y_pred
        dl_dw += -2 * X[i] * error
        dl_db += -2 * error
    w = w - (1 / N) * dl_dw * alpha
    b = b - (1 / N) * dl_db * alpha
    return w, b

def avg_loss(X, y, w, b):
    N = len(X)
    total_error = 0.0
    for i in range(N):
        y_pred = np.dot(X[i], w) + b
        total_error += (y[i] - y_pred) ** 2
    return total_error / float(N)

def train_and_plot(X, y, w, b, alpha, epochs):
    loss_history = []
    for e in range(epochs):
        w, b = update_w_and_b(X, y, w, b, alpha)
        loss = avg_loss(X, y, w, b)
        loss_history.append(loss)
        if e % 100 == 0:
            print(f"Época {e} | Pérdida: {loss:.4f} | w:{w}, b:{b:.4f}")
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), loss_history, label='Pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title('Progreso del Entrenamiento de Regresión Lineal')
    plt.legend()
    plt.grid(True)
    plt.show()
    return w, b

def predict(x, w, b):
    return np.dot(x, w) + b

class LinearRegressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Regresión Lineal GUI")
        self.root.geometry("500x300")
        self.w = None
        self.b = None
        self.alpha = 0.01
        self.epochs = 1000

        tk.Label(root, text="Regresión Lineal - Predicción de Admisión", font=("Arial", 14)).pack(pady=10)
        tk.Label(root, text="Tasa de aprendizaje (alpha):").pack(pady=5)
        
        self.alpha_entry = tk.Entry(root)
        self.alpha_entry.insert(0, "0.01")
        self.alpha_entry.pack(pady=5)
        
        tk.Label(root, text="Número de épocas:").pack(pady=5)
        self.epochs_entry = tk.Entry(root)
        self.epochs_entry.insert(0, "1000")
        self.epochs_entry.pack(pady=5)

        tk.Button(root, text="Cargar Dataset", command=self.load_data).pack(pady=5)
        tk.Button(root, text="Entrenar Modelo", command=self.train_model).pack(pady=5)
        tk.Button(root, text="Predecir", command=self.predict).pack(pady=5)

    def load_data(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.data = pd.read_csv(filepath)
            self.prepare_data()
            messagebox.showinfo("Información", "Dataset cargado exitosamente.")

    def prepare_data(self):
        self.X = self.data[['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']].values
        self.y = self.data['Chance of Admit '].values
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)

    def train_model(self):
        if hasattr(self, 'X') and hasattr(self, 'y'):
            self.alpha = float(self.alpha_entry.get())
            self.epochs = int(self.epochs_entry.get())
            self.w, self.b = np.zeros(self.X.shape[1]), 0.0
            self.w, self.b = train_and_plot(self.X, self.y, self.w, self.b, self.alpha, self.epochs)
            messagebox.showinfo("Información", "Modelo entrenado exitosamente.")
        else:
            messagebox.showwarning("Advertencia", "Primero cargue el dataset.")

    def predict(self):
        if self.w is not None and self.b is not None:
            x_new = self.X[0]
            y_pred = predict(x_new, self.w, self.b)
            messagebox.showinfo("Predicción", f"Predicción: {y_pred:.4f}")
        else:
            messagebox.showwarning("Advertencia", "Primero entrene el modelo.")

# Crear la ventana principal
root = tk.Tk()
app = LinearRegressionGUI(root)
root.mainloop()
