import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist #type:ignore 
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import Dense, Flatten #type:ignore
from tensorflow.keras.utils import to_categorical #type:ignore

# Cargar el dataset MNIST
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Normalizar los valores de los píxeles a un rango de 0 a 1
train_X = train_X / 255.0
test_X = test_X / 255.0

# Convertir las etiquetas a formato one-hot
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

# Diseñar la red neuronal
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Aplanar las imágenes de 28x28 a un vector de 784 elementos
    Dense(128, activation='relu'),  # Capa densa con 128 neuronas y activación ReLU
    Dense(64, activation='relu'),   # Capa densa con 64 neuronas y activación ReLU
    Dense(10, activation='softmax') # Capa de salida con 10 neuronas (una para cada dígito) y activación softmax
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_X, train_y, epochs=10, batch_size=32, validation_data=(test_X, test_y))

# Evaluar el modelo
test_loss, test_acc = model.evaluate(test_X, test_y)
print(f'Precisión en el conjunto de prueba: {test_acc}')

# Visualización de resultados
# Gráfico de precisión
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Gráfico de pérdida
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Clasificación de una imagen del conjunto de prueba
index = 5  
image = test_X[index]

# Mostrar la imagen
plt.imshow(image, cmap='gray')
plt.title(f'Etiqueta Verdadera: {np.argmax(test_y[index])}')
plt.show()

# Predecir la etiqueta de la imagen
prediction = model.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(prediction)

print(f'Predicción del Modelo: {predicted_label}')
