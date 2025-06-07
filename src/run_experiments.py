# src/run_experiments.py

import argparse
import pandas as pd
from data_processing import preprocess_dataset
from models import usar_cblof, usar_iforest, usar_ecod, usar_autoencoder, usar_hbos, usar_mcd, usar_vae
import os
from config import DATASETS, OUTPUT_DIR, SHAP_DIR, LIME_DIR, PDP_DIR
from xai_utils import (usar_shap_global,usar_lime, usar_pdp)
from evaluation import representar_fallos
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos de XAI para PdM")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices={'metro', 'hydraulic'},
                        help="Nombre del dataset a usar: hydraulic o metro")
    parser.add_argument("-e", "--experiment", type=str, required=True,
                        choices=["benchmark", "metricas", "representar_fallos", "shap", "lime", "pdp"],
                        help="Tipo de experimento a ejecutar")
    parser.add_argument("-s", "--seeds", type=int, default=1, help="Numero de semillas a ejectuar")
    parser.add_argument("-p", "--plot", action="store_true", help="Mostrar gráficas o no")
    parser.add_argument("-t", "--evaluation", type=str, default="test", choices=["train", "test", "both"],
                        help="Conjunto de datos a usar para la evaluación del experimento")
    
    args = parser.parse_args()
    dataset_name = args.dataset
    experiment_type = args.experiment
    n_seeds = args.seeds
    show_plot = args.plot
    evaluation_type = args.evaluation

    # Creamos las carpetas de salida si no existen
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if experiment_type == "shap":
        os.makedirs(SHAP_DIR, exist_ok=True)
        os.makedirs(os.path.join(SHAP_DIR, dataset_name), exist_ok=True)

    if experiment_type == "lime":
        os.makedirs(LIME_DIR, exist_ok=True)
        os.makedirs(os.path.join(LIME_DIR, dataset_name), exist_ok=True)

    if experiment_type == "pdp":
        os.makedirs(PDP_DIR, exist_ok=True)
        os.makedirs(os.path.join(PDP_DIR, dataset_name), exist_ok=True)
        

    X_train, y_train, X_test, y_test, X_train_norm, X_test_norm, anomalias_fraccion = preprocess_dataset(dataset_name, show_plot)
    
    modelos = {
        'CBLOF': usar_cblof,
        'IForest': usar_iforest,
        'ECOD': usar_ecod,
        'AutoEncoder': usar_autoencoder,
        'HBOS': usar_hbos,
        'MCD': usar_mcd,
        'VAE': usar_vae,
    }

    importances = pd.DataFrame()

    if evaluation_type == "both":
        metrics_type = ["train", "test"]
    else:
        metrics_type = [evaluation_type]

    for type in metrics_type:

        metrics_df = pd.DataFrame(columns=['Modelo', 'Semilla', 'Normalizado', 'Contaminacion', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'F1-score', 'Sensibilidad', 'Especificidad', 'Precisión', 'ROC-AUC', 'Tiempo (s)'])
        
        for model_name, model_function in modelos.items():
            print(f'\n********** {model_name} **********')    
            for seed in range(n_seeds):
                print(f'\nSemilla: {seed}')
                # Entrenar modelo
                if type == "train":
                    X_ev = X_train.copy()
                    X_ev_norm = X_train_norm.copy()
                    y_ev = y_train.copy()
                elif type == "test":
                    X_ev = X_test.copy()
                    X_ev_norm = X_test_norm.copy()
                    y_ev = y_test.copy()

                if experiment_type == "benchmark":
                    metrics_df, model = model_function(X_train, y_train, X_ev, y_ev, metrics_df, False, None, seed)
                    metrics_df, model = model_function(X_train_norm, y_train, X_ev_norm, y_ev, metrics_df, True, None, seed)
                    metrics_df, model = model_function(X_train, y_train, X_ev, y_ev, metrics_df, False, anomalias_fraccion, seed)
                    metrics_df, model = model_function(X_train_norm, y_train, X_ev_norm, y_ev, metrics_df, True, anomalias_fraccion, seed)
                elif dataset_name == "metro":
                    metrics_df, model = model_function(X_train_norm, y_train, X_ev_norm, y_ev, metrics_df, True, None, seed)
                elif dataset_name == "hydraulic":
                    metrics_df, model = model_function(X_train_norm, y_train, X_ev_norm, y_ev, metrics_df, True, anomalias_fraccion, seed)

                if experiment_type == "representar_fallos":
                    if dataset_name == "metro" or dataset_name == "hydraulic":
                        representar_fallos(model, model_name, dataset_name, X_ev_norm, y_ev)

                # SHAP
                if experiment_type == "shap":
                    if dataset_name == "metro" or dataset_name == "hydraulic":
                        importances = usar_shap_global(model, model_name, dataset_name, X_train_norm, X_ev_norm, importances, show_plot)

                # LIME
                if experiment_type == "lime":
                    if dataset_name == "metro" or dataset_name == "hydraulic":
                        usar_lime(model, model_name, dataset_name, X_train_norm, X_ev_norm)

                # PDP
                if experiment_type == "pdp":
                    if dataset_name == "metro" or dataset_name == "hydraulic":
                        usar_pdp(model, model_name, dataset_name, X_train_norm, y_train, show_plot)

                if model_name == 'ECOD' or model_name == "HBOS": # No dependen de aleatoriedad
                    break

        if experiment_type == "benchmark" or experiment_type == "metrics":
            metrics_df.to_excel(fr'{OUTPUT_DIR}/resultado_{experiment_type}_{dataset_name}_{type}.xlsx', index=False)

    # SHAP
    if experiment_type == "shap":
        # Ponemos 'Modelo' como primera columna
        cols = importances.columns.tolist()
        cols = ['Modelo'] + [col for col in cols if col != 'Modelo']
        importances = importances[cols]

        importances.to_excel(os.path.join(OUTPUT_DIR, fr'{dataset_name}_shap_rankings.xlsx'), index=False)




if __name__ == "__main__":
    main()
