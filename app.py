import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os

# Definir la ruta de las carpetas de geodas, rosa del desierto y turmalina negra
ruta_geodas = r"train\Geoda"
ruta_rosa_del_desierto = r"train\Rosa del Desierto"
ruta_turmalina_negra = r"train\Turmalina Negra"
height, width = 28, 28  # Tamaño deseado para las imágenes

def cargar_imagenes(ruta_geodas, ruta_rosa_del_desierto, ruta_turmalina_negra, height, width):
    imagenes = []
    etiquetas = []

    # Cargar imágenes de geodas
    for archivo in os.listdir(ruta_geodas):
        ruta_imagen = os.path.join(ruta_geodas, archivo)
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen is not None:
            imagen_redimensionada = cv2.resize(imagen, (height, width)) / 255.0  # Normalizar entre 0 y 1
            imagenes.append(imagen_redimensionada)
            etiquetas.append([1, 0, 0])  # Etiqueta [1, 0, 0] para geodas

    # Cargar imágenes de rosa del desierto
    for archivo in os.listdir(ruta_rosa_del_desierto):
        ruta_imagen = os.path.join(ruta_rosa_del_desierto, archivo)
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen is not None:
            imagen_redimensionada = cv2.resize(imagen, (height, width)) / 255.0  # Normalizar entre 0 y 1
            imagenes.append(imagen_redimensionada)
            etiquetas.append([0, 1, 0])  # Etiqueta [0, 1, 0] para rosa del desierto

    # Cargar imágenes de turmalina negra
    for archivo in os.listdir(ruta_turmalina_negra):
        ruta_imagen = os.path.join(ruta_turmalina_negra, archivo)
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if imagen is not None:
            imagen_redimensionada = cv2.resize(imagen, (height, width)) / 255.0  # Normalizar entre 0 y 1
            imagenes.append(imagen_redimensionada)
            etiquetas.append([0, 0, 1])  # Etiqueta [0, 0, 1] para turmalina negra

    return np.array(imagenes), np.array(etiquetas)

# Cargar imágenes de geodas, rosa del desierto y turmalina negra
X_data, y_data = cargar_imagenes(ruta_geodas, ruta_rosa_del_desierto, ruta_turmalina_negra, height, width)

# Definir la arquitectura de la red neuronal
model = models.Sequential([
    layers.Flatten(input_shape=(height, width)),  # Capa de entrada para imágenes
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # Salida con tres clases
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_data, y_data, epochs=10)

# Cargar la imagen específica que quieres clasificar
ruta_imagen_a_clasificar = r"train\prueba.jpg"
imagen_a_clasificar = cv2.imread(ruta_imagen_a_clasificar, cv2.IMREAD_GRAYSCALE)
if imagen_a_clasificar is not None:
    imagen_redimensionada = cv2.resize(imagen_a_clasificar, (height, width))
    imagen_redimensionada = np.expand_dims(imagen_redimensionada, axis=0)  # Agregar dimensión de lote
    prediccion = model.predict(imagen_redimensionada)
    clase_predicha = np.argmax(prediccion)  # Obtener la clase con mayor probabilidad
    probabilidad_predicha = np.max(prediccion)  # Obtener la probabilidad más alta

if clase_predicha == 0:
    probabilidad_geoda = probabilidad_predicha / np.sum(prediccion)
    print(f"La imagen es una geoda con un {probabilidad_geoda * 100:.2f}% de probabilidad.")
elif clase_predicha == 1:
    probabilidad_rosa = probabilidad_predicha / np.sum(prediccion)
    print(f"La imagen es una rosa del desierto con un {probabilidad_rosa * 100:.2f}% de probabilidad.")
elif clase_predicha == 2:
    probabilidad_turmalina = probabilidad_predicha / np.sum(prediccion)
    print(f"La imagen es una turmalina negra con un {probabilidad_turmalina * 100:.2f}% de probabilidad.")
else:
    print(f"La imagen es de otro tipo de piedra.")
