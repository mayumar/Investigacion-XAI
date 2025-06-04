# src/run_experiments.py

import argparse
import pandas as pd
from data_processing import preprocess_dataset
from models import usar_cblof, usar_iforest, usar_ecod, usar_autoencoder, usar_hbos, usar_mcd, usar_vae
import os
# from config import DATASETS, OUTPUT_DIR, RANDOM_STATE
# from xai_utils import (shap_global_importances,
#                        lime_local_explanation,
#                        pdp_plot)
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos de XAI para PdM")
    parser.add_argument("--dataset", type=str, required=True, choices={'metro', 'hydraulic'},
                        help="Nombre del dataset a usar: hydraulic o metro")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["all_metrics", "get_metrics", "shap", "lime", "pdp", "all_xai"],
                        help="Tipo de experimento a ejecutar")
    parser.add_argument("--seeds", type=int, default=1, help="Numero de semillas a ejectuar")
    # TODO: en metrics dar la opción de hacerlo con test o con train
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUTPUT_DIR = os.path.join(BASE_DIR, "results")
    
    args = parser.parse_args()
    dataset_name = args.dataset
    experiment_type = args.experiment
    n_seeds = args.seeds
    
    X_train, y_train, X_test, y_test, X_train_norm, X_test_norm, anomalias_fraccion = preprocess_dataset(dataset_name)

    modelos = {
        'CBLOF': usar_cblof,
        'IForest': usar_iforest,
        'ECOD': usar_ecod,
        'AutoEncoder': usar_autoencoder,
        'HBOS': usar_hbos,
        'MCD': usar_mcd,
        'VAE': usar_vae,
    }

    metrics_df = pd.DataFrame(columns=['Modelo', 'Semilla', 'Normalizado', 'Contaminacion', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'F1-score', 'Sensibilidad', 'Especificidad', 'Precisión', 'ROC-AUC', 'Tiempo (s)'])

    importances = pd.DataFrame()

    for model_name, model_function in modelos.items():
        print(f'\n********** {model_name} **********')    
        for seed in range(n_seeds):
            print(f'\nSemilla: {seed}')
            # Entrenar modelo
            if experiment_type == "all_metrics":
                metrics_df, model = model_function(X_train, y_train, X_test, y_test, metrics_df, False, None, seed)
                metrics_df, model = model_function(X_train_norm, y_train, X_test_norm, y_test, metrics_df, True, None, seed) # -> El que mejor va en HBOS
                metrics_df, model = model_function(X_train, y_train, X_test, y_test, metrics_df, False, anomalias_fraccion, seed)
                metrics_df, model = model_function(X_train_norm, y_train, X_test_norm, y_test, metrics_df, True, anomalias_fraccion, seed)
            elif dataset_name == "metro":
                metrics_df, model = model_function(X_train_norm, y_train, X_test_norm, y_test, metrics_df, True, None, seed)

            # SHAP
            if experiment_type == "shap":
                print("shap")
                # importances = usar_shap_global(model, model_name, X_train_norm, X_test_norm, importances, True)
                # importances.to_excel(fr'rankings_after_{model_name}.xlsx', index=False)

            # LIME
            if experiment_type == "lime":
                print("lime")
                # observations = [
                # TODO: Rellenar aquí para echar lime a andar
                # ]
                # usar_lime(model, model_name, X_train_norm, X_test_norm, observations)

            # PDP
            if experiment_type == "pdp":
                print("pdp")
                # usar_pdp(model, model_name, X_train_norm, y_train, showPlot=True)

            if model_name == 'ECOD' or model_name == "HBOS":
                break

    if experiment_type == "all_metrics" or experiment_type == "get_metrics":
        metrics_df.to_excel(fr'{OUTPUT_DIR}/resultado_metricas_{dataset_name}.xlsx', index=False)

if __name__ == "__main__":
    main()
