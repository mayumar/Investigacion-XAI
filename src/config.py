# src/config.py

import os
from datetime import timedelta

# ======================
# RUTAS GENERALES
# ======================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

RANDOM_STATE = 42
TEST_SIZE = 0.4   # para datasets sin división temporal
CONTAMINATION = None  # si queremos forzar una fracción fija de anomalías

# ======================
# PARÁMETROS ESPECÍFICOS POR DATASET
# ======================
DATASETS = {
    "hydraulic": {
        "raw_folder": os.path.join(DATA_RAW_DIR, "hydraulic"),
        "processed_folder": os.path.join(DATA_PROCESSED_DIR, "hydraulic"),
        # Qué columnas eliminar tras el análisis de correlación
        "drop_features": ['PS2', 'FS1', 'SE', 'PS6', 'CE', 'CP', 'FS2', 
                          'TS1', 'TS2', 'TS3', 'TS4'],
        # Cómo añadimos la columna Target (en hydraulic: cooler == 3 → Target=1)
        "label_method": "hydraulic_profile",  
    },
    "metro": {
        "raw_folder": os.path.join(DATA_RAW_DIR, "metro"),
        "processed_folder": os.path.join(DATA_PROCESSED_DIR, "metro"),
        # Columnas a eliminar para Metro (según tu exploración: DV_eletric, TP3, H1, COMP, MPG)
        "drop_features": ['DV_eletric', 'TP3', 'H1', 'COMP', 'MPG'],
        # Método para etiquetar (“failure” basado en intervalos temporales)
        "label_method": "metro_intervals",
        # Parámetros adicionales para metro, p.ej. lista de intervalos
        "metro_intervals": [
            # (inicio, fin) de cada intervalo anómalo
            ("2020-04-12 11:50:00", "2020-04-12 23:30:00"),
            ("2020-04-18 00:00:00", "2020-04-19 01:30:00"),
            # ... resto de intervalos que ya tienes en tu código
        ],
        # Frecuencia de resampleo para metro (por ejemplo, "30T" = 30 minutos)
        "resample_freq": "30T",
    },
}
