#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ============================================
# CONFIGURACIÓN
# ============================================
IMG_FOLDER = '/zfs-home/202503037C/Face/img_align_celeba/'
ATTR_FILE = '/zfs-home/202503037C/Face/list_attr_celeba.csv'
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 25
NUM_ATTR = 40
AUTOTUNE = tf.data.AUTOTUNE

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
# CARGAR DATAFRAME
# ============================================
print("\nCargando CSV")
df = pd.read_csv(ATTR_FILE)
print(f"Total imágenes en CSV: {len(df)}")

# ============================================
# DIVIDIR DATOS
# ============================================
print("\n📊 Dividiendo datos...")
indices = np.arange(len(df))
idx_temp, idx_test = train_test_split(indices, test_size=0.2, random_state=42)
idx_train, idx_val = train_test_split(idx_temp, test_size=0.2, random_state=42)

df_train = df.iloc[idx_train].reset_index(drop=True)
df_val = df.iloc[idx_val].reset_index(drop=True)
df_test = df.iloc[idx_test].reset_index(drop=True)

print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

# ============================================
# FUNCIÓN DE CARGA CON tf.data
# ============================================
def process_image(image_id, attributes, img_folder=IMG_FOLDER, img_size=IMG_SIZE):
    img_path = tf.strings.join([img_folder, image_id])
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img, attributes

def create_dataset(df, img_folder, batch_size, img_size, shuffle=True):
    attr_cols = df.columns[1:].tolist()
    attr_values = df[attr_cols].values.astype(np.float32)
    attr_values = (attr_values + 1) / 2  # -1/1 -> 0/1
    
    dataset = tf.data.Dataset.from_tensor_slices((
        df['image_id'].values,
        attr_values
    ))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    
    dataset = dataset.map(
        lambda img_id, attrs: process_image(img_id, attrs, img_folder, img_size),
        num_parallel_calls=AUTOTUNE
    )
    
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

print("\n🔄 Creando datasets con tf.data...")
train_ds = create_dataset(df_train, IMG_FOLDER, BATCH_SIZE, IMG_SIZE, shuffle=True)
val_ds = create_dataset(df_val, IMG_FOLDER, BATCH_SIZE, IMG_SIZE, shuffle=False)
test_ds = create_dataset(df_test, IMG_FOLDER, BATCH_SIZE, IMG_SIZE, shuffle=False)

print("Datasets listos")

# ============================================
# MODELO GRANDE (más potente)
# ============================================
def crear_modelo_grande():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Bloque 1: 64 filtros
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Bloque 2: 128 filtros
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Bloque 3: 256 filtros
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Bloque 4: 512 filtros (nuevo)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Cabezal clasificador
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(NUM_ATTR, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs)

modelo = crear_modelo_grande()
modelo.compile(
    optimizer=keras.optimizers.Adam(0.0005),  # LR más baja para modelo grande
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)
modelo.summary()

# ============================================
# CALLBACKS 
# ============================================
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'celebA_grande_mejor.h5', 
        save_best_only=True,
        monitor='val_binary_accuracy',
        mode='max'
    ),
    keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6)
]

# ============================================
# ENTRENAR
# ============================================
print("\nIniciando entrenamiento (modelo grande)...")
history = modelo.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ============================================
# GUARDAR MODELO FINAL (con otro nombre)
# ============================================
modelo.save('celebA_modelo_grande_final.h5')
print("\n Modelo grande guardado como 'celebA_modelo_grande_final.h5'")

# Guardar también las métricas
import json
with open('historial_grande.json', 'w') as f:
    json.dump(history.history, f)
print("Historial guardado en 'historial_grande.json'")
