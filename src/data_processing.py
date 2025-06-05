# src/data_processing.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from config import DATASETS, RANDOM_STATE, TEST_SIZE

def load_hydraulic():
    """
    Carga todos los .txt de sensores y profile.txt, 
    calcula medias por fila y crea columna 'Target'.
    """
    folder = DATASETS["hydraulic"]["raw_folder"]
    sensor_files = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6',
                    'EPS1', 'FS1', 'FS2',
                    'TS1', 'TS2', 'TS3', 'TS4',
                    'VS1', 'CE', 'CP', 'SE']
    medias = {}
    for sensor in sensor_files:
        path_sensor = os.path.join(folder, f"{sensor}.txt")
        df_sensor = pd.read_csv(path_sensor, sep="\t", header=None)
        medias[sensor] = df_sensor.mean(axis=1)
    df_medias = pd.DataFrame(medias)

    # Leer profile.txt
    path_profile = os.path.join(folder, "profile.txt")
    df_profile = pd.read_csv(path_profile, sep="\t", header=None,
                             names=['cooler', 'valve', 'leakage', 'accumulator', 'stable'])

    # Definición de la etiqueta: cooler == 3 → Target = 1 (óptimo), else 0
    optimal = (df_profile['cooler'] == 3)
    df_medias['Target'] = optimal.astype(int)

    return df_medias  # DataFrame con columnas de sensores + Target

def load_data_metro():
    print("--- Procesando dataset MetroPT-3 ---")
    df = pd.read_csv(DATASETS["metro"]["ruta_origen"])
    df.drop(df.columns[0], axis=1, inplace=True)

    df = df.drop(DATASETS["metro"]["variables_eliminar"], axis=1, inplace=False) # Aqui se han quedado Reservoirs y TP2

    # Nos aseguramos de que 'timestamp' es de tipo datetime y el índice
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Remuestrea a intervalos de 1 hora, aplicando una función agregada (por ejemplo, la media)
    df = df.asfreq(DATASETS["metro"]["resample_freq"], method=DATASETS["metro"]["fill_method"])

    # Se ordena el dataset cronologicamente
    df = df.sort_index()

    # Inicializamos la columna binaria en 0
    df['failure'] = 0

    # Iteramos sobre cada intervalo y asignamos 1 a los rangos correspondientes
    for inicio, fin in DATASETS["metro"]["intervalos"]:
        df.loc[(df.index >= inicio) & (df.index <= fin), 'failure'] = 1


    # Calcular el índice de división para obtener el 60% de los datos
    split_index = int((1 - TEST_SIZE) * len(df))

    # Dividir en conjuntos de entrenamiento y prueba
    train_dataset = df.iloc[:split_index]
    test_dataset = df.iloc[split_index:]

    X_train = train_dataset.drop('failure', axis=1, inplace=False)
    y_train = train_dataset['failure']

    X_test = test_dataset.drop('failure', axis=1, inplace=False)
    y_test = test_dataset['failure']

    return X_train, X_test, y_train, y_test

def preprocess_dataset(dataset_name):
    # 1) Carga específica según dataset_name
    if dataset_name == "metro":
        X_train, X_test, y_train, y_test = load_data_metro()
    else:
        raise ValueError(f"Dataset desconocido: {dataset_name}")
    
    # Fracción de anomalías que tiene el conjunto de datos
    anomalias_fraccion = np.count_nonzero(y_train.values) / len(y_train.values)
    print("Fracción de anomalías:", anomalias_fraccion)

    anomalias_porcentaje = round(anomalias_fraccion * 100, ndigits=4)
    print("Porcentaje de anomalías:", anomalias_porcentaje)

    # Normalizamos los datos en el itervalo [0, 1]
    scaler = MinMaxScaler()

    X_train_norm = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test),   # <-- transform, no fit_transform
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train, y_train, X_test, y_test, X_train_norm, X_test_norm, anomalias_fraccion
