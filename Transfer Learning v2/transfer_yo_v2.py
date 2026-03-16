#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import sys

# ============================================
# CONFIGURACIÓN
# ============================================
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS_FASE1 = 20
EPOCHS_FASE2 = 10
NUM_ATTR = 40  # del modelo original (no lo usaremos directamente)

# Rutas
MODELO_PRETRAINED = '/zfs-home/202503037C/Face/modelo_potente/celebA_grande_mejor.h5'
TRAIN_DIR = '/zfs-home/202503037C/Face/reconocimiento/train'
VAL_DIR = '/zfs-home/202503037C/Face/reconocimiento/validation'

# ============================================
# MEMORIA DINÁMICA GPU
# ============================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Memoria dinámica activada en {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

# ============================================
# 1. CARGAR MODELO PRE-ENTRENADO
# ============================================
print("\nCargando modelo pre-entrenado")
modelo_completo = keras.models.load_model(MODELO_PRETRAINED)
print("Modelo cargado")

# ============================================
# 2. EXTRAER CAPAS CONVOLUCIONALES (SIN CLASIFICADOR)
# ============================================
# El clasificador original comienza después de GlobalAveragePooling2D
# Identificamos el índice de esa capa
for i, layer in enumerate(modelo_completo.layers):
    if isinstance(layer, layers.GlobalAveragePooling2D):
        idx_gap = i
        print(f"   Capa GAP encontrada en índice {i}: {layer.name}")
        break

# Tomamos todas las capas hasta antes de GAP (solo convolucionales + pooling)
conv_base = keras.Model(
    inputs=modelo_completo.input,
    outputs=modelo_completo.layers[idx_gap - 1].output  # salida antes de GAP
)

print("\nArquitectura base convolucional:")
conv_base.summary()

# ============================================
# 3. CONGELAR CAPAS PRE-ENTRENADAS 
# ============================================
conv_base.trainable = False
print("Capas convolucionales congeladas")

# ============================================
# 4. CONSTRUIR NUEVO MODELO CON CLASIFICADOR BINARIO 
# ============================================
model = keras.Sequential([
    conv_base,
    layers.GlobalAveragePooling2D(name='gap_nuevo'),
    layers.Dropout(0.5, name='dropout_nuevo_1'),
    layers.Dense(128, activation='relu', name='fc_nuevo_1'),
    layers.Dropout(0.3, name='dropout_nuevo_2'),
    layers.Dense(1, activation='sigmoid', name='salida_binaria')  # una neurona: yo(1) / otros(0)
], name='Modelo_Reconocimiento_Facial')

print("\nModelo completo (base congelada + nuevo clasificador):")
model.summary()

# ============================================
# 5. COMPILAR MODELO
# ============================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# ============================================
# 6. DATA AUGMENTATION
# ============================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
# Para validación, solo normalizar (sin aumento)
val_datagen = ImageDataGenerator(rescale=1./255)

# Crear generadores
print("\n🔄 Creando generadores con aumento de datos...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['otros', 'yo'],  # otros=0, yo=1
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['otros', 'yo'],
    shuffle=False
)

print(f"\nClases mapeadas: {train_generator.class_indices}")


# Calcular pesos para balancear clases
total_train = train_generator.samples
total_yo = sum(train_generator.classes == 1)  
total_otros = total_train - total_yo

weight_for_yo = (1 / total_yo) * total_train / 2.0
weight_for_otros = (1 / total_otros) * total_train / 2.0

class_weight = {1: weight_for_yo, 0: weight_for_otros}
print(f"\nPesos de clase:")
print(f"   Yo (clase 1): {weight_for_yo:.2f}")
print(f"   Otros (clase 0): {weight_for_otros:.2f}")
# ============================================
# 7. FASE 1: ENTRENAR SOLO EL NUEVO CLASIFICADOR
# # ============================================
print("\nFASE 1: Entrenando solo el nuevo clasificador")
print(f"   Train batches: {len(train_generator)}")
print(f"   Val batches: {len(validation_generator)}")

callbacks_fase1 = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(
        'mejor_modelo_fase1.h5',
        save_best_only=True,
        monitor='val_accuracy'
    )
]

history_fase1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_FASE1,
    callbacks=callbacks_fase1,
    class_weight=class_weight,  #PESO DE CLASES PARA FASE 1
    verbose=1
)

# ============================================
# 8. FASE 2: FINE-TUNING 
# ============================================
print("\nFASE 2: Fine-tuning (descongelar últimas capas)")

# Descongelar las últimas capas convolucionales
conv_base.trainable = True

# Congelar las primeras capas para no arruinar lo aprendido
# Por ejemplo, congelar las primeras 10 capas
for layer in conv_base.layers[:10]:
    layer.trainable = False

print("\nCapas entrenables después de fine-tuning:")
for i, layer in enumerate(model.layers[0].layers):
    if layer.trainable:
        print(f"   Capa {i}: {layer.name} (entrenable)")
    else:
        print(f"   Capa {i}: {layer.name} (congelada)")

# Recompilar con learning rate más baja
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

callbacks_fase2 = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(
        'mejor_modelo_fase2.h5',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
]

history_fase2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS_FASE2,
    callbacks=callbacks_fase2,
    class_weight=class_weight,  # <-- peso de clases también para fase 2
    verbose=1
)

# ============================================
# 9. GUARDAR MODELO FINAL
# ============================================
model.save('modelo_reconocimiento_final.h5')
print("\nModelo final guardado como 'modelo_reconocimiento_final.h5")

# Guardar historiales
import pickle
with open('historial_fase1.pkl', 'wb') as f:
    pickle.dump(history_fase1.history, f)
with open('historial_fase2.pkl', 'wb') as f:
    pickle.dump(history_fase2.history, f)

print("Historiales guardados")
