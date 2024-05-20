import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os

# Definir la ruta de las carpetas de geodas y rosa del desierto
ruta_geodas = r"train\Geoda"
ruta_rosa_del_desierto = r"train\Rosa del Desierto"
height, width = 28, 28  # Tamaño deseado para las imágenes

# Función para cargar imágenes de geodas y rosa del desierto
def cargar_imagenes(ruta_geodas, ruta_rosa_del_desierto, height, width):
    imagenes = []
    etiquetas = []

    # Cargar imágenes de geodas
    for archivo in os.listdir(ruta_geodas):
        ruta_imagen = os.path.join(ruta_geodas, archivo)
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen is not None:
            imagen_redimensionada = cv2.resize(imagen, (height, width))
            imagenes.append(imagen_redimensionada)
            etiquetas.append(0)  # Etiqueta 0 para geodas

    # Cargar imágenes de rosa del desierto
    for archivo in os.listdir(ruta_rosa_del_desierto):
        ruta_imagen = os.path.join(ruta_rosa_del_desierto, archivo)
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen is not None:
            imagen_redimensionada = cv2.resize(imagen, (height, width))
            imagenes.append(imagen_redimensionada)
            etiquetas.append(1)  # Etiqueta 1 para rosa del desierto

    return np.array(imagenes), np.array(etiquetas)

# Cargar imágenes de geodas y rosa del desierto
X_data, y_data = cargar_imagenes(ruta_geodas, ruta_rosa_del_desierto, height, width)

# Definir la arquitectura de la red neuronal
model = models.Sequential([
    layers.Flatten(input_shape=(height, width)),  # Capa de entrada para imágenes
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Salida binaria
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_data, y_data, epochs=10)

# Cargar la imagen específica que quieres clasificar
ruta_imagen_a_clasificar = r"train\prueba.jpg"
imagen_a_clasificar = cv2.imread(ruta_imagen_a_clasificar, cv2.IMREAD_GRAYSCALE)
if imagen_a_clasificar is not None:
    imagen_redimensionada = cv2.resize(imagen_a_clasificar, (height, width))
    imagen_redimensionada = np.expand_dims(imagen_redimensionada, axis=0)  # Agregar dimensión de lote
    prediccion = model.predict(imagen_redimensionada)
    if prediccion > 0.5:
        print("La imagen es una rosa del desierto.")
    else:
        print("La imagen es una geoda.")
else:
    print("Error al cargar la imagen.")