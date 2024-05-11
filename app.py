import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os

# Ruta del directorio de las imágenes de geodas
ruta_directorio = "/Rocas/train/Geoda"
height, width = 28, 28  # Tamaño deseado para las imágenes

# Función para cargar imágenes de geodas
def cargar_imagenes_geodas(ruta_directorio, height, width):
    imagenes = []
    etiquetas = []

    # Obtener la lista de archivos en el directorio
    archivos = os.listdir(ruta_directorio)

    for archivo in archivos:
        # Comprobar si el archivo es una imagen
        if archivo.endswith(".jpg") or archivo.endswith(".png"):
            # Cargar la imagen usando OpenCV
            imagen = cv2.imread(os.path.join(ruta_directorio, archivo), cv2.IMREAD_GRAYSCALE)
            # Redimensionar la imagen si es necesario
            imagen_redimensionada = cv2.resize(imagen, (height, width))
            # Agregar la imagen al arreglo de imágenes
            imagenes.append(imagen_redimensionada)
            # Agregar la etiqueta correspondiente (en este caso, todas las imágenes son de geodas, etiqueta 1)
            etiquetas.append(1)

    # Convertir las listas de imágenes y etiquetas a arreglos numpy
    imagenes_np = np.array(imagenes)
    etiquetas_np = np.array(etiquetas)

    return imagenes_np, etiquetas_np

# Cargar las imágenes de geodas en un arreglo de NumPy
X_geodes, y_geodes = cargar_imagenes_geodas(ruta_directorio, height, width)

# Imprimir las formas de X_geodes e y_geodes para verificar
print("Forma de X_geodes:", X_geodes.shape)  # Debería ser (cantidad_imágenes, height, width)
print("Forma de y_geodes:", y_geodes.shape)  # Debería ser (cantidad_imágenes,)

# Definir la arquitectura de la red neuronal
model = models.Sequential([
    layers.Flatten(input_shape=(height, width)),  # Capa de entrada para imágenes de geodas
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Salida binaria para geodas
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dividir datos en entrenamiento y validación
X_train, X_valid = X_geodes[:50], X_geodes[50:]
y_train, y_valid = y_geodes[:50], y_geodes[50:]

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))

# Hacer predicciones
predictions = model.predict(X_valid)

# Establecer un umbral de confianza
umbral_confianza = 0.5  # Puedes ajustar este valor según la confianza deseada

# Mostrar los resultados
for prediction in predictions:
    if prediction > umbral_confianza:
        print("Es una Geoda. Una geoda es una cavidad rocosa, normalmente cerrada, y totalmente tapizada con cristales y otras materias minerales. No es realmente un mineral sino una composición de formaciones magmáticas, cristalinas y/o sedimentarias.")
    else:
        print("No es una Geoda.")
