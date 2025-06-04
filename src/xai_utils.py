# src/xai_utils.py

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from lime.lime_tabular import LimeTabularExplainer
# import pdpbox.pdp as pdp
from config import OUTPUT_DIR

def usar_shap_global(clf, clf_name, X_train, X_test, importances_df, show_plot=True, failure_type=None):
    # Crear un SHAP explainer
    explainer = shap.Explainer(clf.predict, X_train)

    # Explicar los valores
    shap_values = explainer(X_test)

    # Generar los reportes
    shap.plots.beeswarm(shap_values, show=show_plot)
    fig = plt.gcf()
    fig.savefig(f"shap_{clf_name}.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Extraemos los valores shap
    shap_array = shap_values.values  # <- Accedemos a los valores puros

    # Calculamos la importancia media usando valores absolutos
    shap_importance = np.abs(shap_array).mean(axis=0)

    # Creamos una fila para este modelo
    model_feature_importance = pd.Series(dict(zip(X_train.columns, shap_importance)))

    # Obtenemos el ranking (1 = más importante)
    model_feature_ranking = model_feature_importance.rank(method='average', ascending=False).astype(int)

    # Añadimos el modelo
    model_feature_ranking['Modelo'] = clf_name

    # Lo añadimos al DataFrame
    importances_df = pd.concat([importances_df, pd.DataFrame([model_feature_ranking])], ignore_index=True)

    return importances_df

def lime_local_explanation(clf, clf_name, X_train, X_test, observations, dataset_name):
    """
    Aplica LIME a las instancias indicadas (índices de filas en X_test),
    guarda los PNG en outputs/figures/<dataset_name>/lime_<clf_name>_inst<ID>.png
    """
    fig_folder = os.path.join(OUTPUT_DIR, "figures", dataset_name)
    os.makedirs(fig_folder, exist_ok=True)

    lime_explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['NoFailure', 'Failure'],
        mode='classification',
        random_state=42
    )

    for example in observations:
        exp = lime_explainer.explain_instance(X_test.loc[example].values,
                                              clf.predict_proba,
                                              num_features=len(X_train.columns))
        fig = exp.as_pyplot_figure()
        # Nombre de archivo con dataset, clasificador e índice de instancia
        fname = os.path.join(fig_folder,
                             f"lime_{clf_name}_inst{str(example).replace(':','-')}.png")
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)

def pdp_plot(clf, clf_name, X_train, y_train, dataset_name, show_plot=True):
    """
    Crea PDP para ciertas variables (ejemplo: 'Motor_current' si existe en X_train).
    Guarda la figura en outputs/figures/<dataset_name>/pdp_<feature>_<clf_name>.png
    """
    fig_folder = os.path.join(OUTPUT_DIR, "figures", dataset_name)
    os.makedirs(fig_folder, exist_ok=True)

    df_combined = pd.concat([X_train, y_train], axis=1)
    # Vamos a usar un ejemplo genérico: si existe 'Motor_current'
    features_a_explorar = [f for f in ['Motor_current'] if f in X_train.columns]

    for feature in features_a_explorar:
        pdp_iso = pdp.PDPIsolate(
            model=clf,
            dataset=df_combined,
            model_features=X_train.columns.tolist(),
            feature=feature,
            num_grid_points=20
        )
        fig, axes = pdp_iso.plot(
            plot_lines=True,
            frac_to_plot=100,
            plot_pts_dist=True
        )
        if show_plot:
            fig.show()
        fname = os.path.join(fig_folder, f"pdp_{feature}_{clf_name}.png")
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
