# src/config.py

import os
from datetime import timedelta
import pandas as pd

# ======================
# RUTAS GENERALES
# ======================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
INPUT_DIR = os.path.join(BASE_DIR, "datasets")

RANDOM_STATE = 42
TEST_SIZE = 0.4   # para datasets sin división temporal
CONTAMINATION = None  # si queremos forzar una fracción fija de anomalías

# ======================
# PARÁMETROS ESPECÍFICOS POR DATASET
# ======================
DATASETS = {
    "metro": {
        "ruta_origen": os.path.join(INPUT_DIR, "MetroPT3(AirCompressor).csv"),
        # Columnas a eliminar despues de correlacion
        "variables_eliminar": ['DV_eletric', 'TP3', 'H1', 'COMP', 'MPG'],
        # Lista de intervalos: cada tupla es (inicio, fin)
        "intervalos": [
            (pd.Timestamp("2020-04-12 11:50:00"), pd.Timestamp("2020-04-12 23:30:00")),# Train
            (pd.Timestamp("2020-04-18 00:00:00"), pd.Timestamp("2020-04-19 01:30:00")),#
            (pd.Timestamp("2020-04-29 03:20:00"), pd.Timestamp("2020-04-29 04:00:00")),
            (pd.Timestamp("2020-04-29 22:00:00"), pd.Timestamp("2020-04-29 22:20:00")),
            (pd.Timestamp("2020-05-13 14:00:00"), pd.Timestamp("2020-05-13 23:59:00")),
            (pd.Timestamp("2020-05-18 05:00:00"), pd.Timestamp("2020-05-18 05:30:00")),
            (pd.Timestamp("2020-05-19 10:10:00"), pd.Timestamp("2020-05-19 11:00:00")),
            (pd.Timestamp("2020-05-19 22:10:00"), pd.Timestamp("2020-05-20 20:00:00")),
            (pd.Timestamp("2020-05-23 09:50:00"), pd.Timestamp("2020-05-23 10:10:00")),
            (pd.Timestamp("2020-05-29 23:30:00"), pd.Timestamp("2020-05-30 06:00:00")),#
            (pd.Timestamp("2020-06-01 15:00:00"), pd.Timestamp("2020-06-01 15:40:00")),
            (pd.Timestamp("2020-06-03 10:00:00"), pd.Timestamp("2020-06-03 11:00:00")),
            (pd.Timestamp("2020-06-05 10:00:00"), pd.Timestamp("2020-06-07 14:30:00")),#
            (pd.Timestamp("2020-07-08 17:30:00"), pd.Timestamp("2020-07-08 19:00:00")),# <------ (1h30) Test empieza aquí
            (pd.Timestamp("2020-07-15 14:30:00"), pd.Timestamp("2020-07-15 19:00:00")),# <------ (4h30)
            (pd.Timestamp("2020-07-17 04:30:00"), pd.Timestamp("2020-07-17 05:30:00"))
        ],
        # Frecuencia de resampleo
        "resample_freq": "30min",
        # Metodo de rellenado
        "fill_method": "ffill",
    },
    "hydraulic": {
        "raw_folder": os.path.join(DATA_RAW_DIR, "hydraulic"),
        "processed_folder": os.path.join(DATA_PROCESSED_DIR, "hydraulic"),
        # Qué columnas eliminar tras el análisis de correlación
        "drop_features": ['PS2', 'FS1', 'SE', 'PS6', 'CE', 'CP', 'FS2', 
                          'TS1', 'TS2', 'TS3', 'TS4'],
        # Cómo añadimos la columna Target (en hydraulic: cooler == 3 → Target=1)
        "label_method": "hydraulic_profile",  
    },
}
