# src/xai_utils.py

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import pdpbox.pdp as pdp
from config import SHAP_DIR, LIME_DIR, PDP_DIR, DATASETS

def usar_shap_global(clf, clf_name, dataset_name, X_train, X_test, importances_df, show_plot=True):
    # Crear un SHAP explainer
    explainer = shap.Explainer(clf.predict, X_train)

    # Explicar los valores
    shap_values = explainer(X_test)

    # Generar los reportes
    shap.plots.beeswarm(shap_values, show=show_plot)

    if not show_plot:
        fig = plt.gcf()
        fig.savefig(os.path.join(os.path.join(SHAP_DIR, dataset_name), f"shap_{clf_name}.png"), bbox_inches='tight', dpi=300)
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

def usar_lime(clf, clf_name, dataset_name, X_train, X_test):
    lime_explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        random_state=42
    )

    for example in DATASETS[dataset_name]["observations"]:
        print(f"Explicando instancia: {example}")
        explanation = lime_explainer.explain_instance(X_test.loc[example], clf.predict_proba, num_features=len(X_train.columns))
        # explanation.show_in_notebook(show_table=True)
        fig = explanation.as_pyplot_figure()
        fig.savefig(os.path.join(os.path.join(LIME_DIR, dataset_name), f"lime_explanation_{str(example).replace(" ", "_").replace(":", "-").replace("/", "-")}_{clf_name}.png"), bbox_inches='tight')


def usar_pdp(clf, clf_name, dataset_name, X_train, y_train, showPlot=True, showGridPoints=False):
  df_combined = pd.concat([X_train, y_train], axis=1)

  for feature in X_train.columns:
    # Creamos el PDP para una característica
    pdp_feature = pdp.PDPIsolate(
        model=clf,
        df=df_combined,
        n_classes=2,
        model_features=X_train.columns,
        feature=feature,
        feature_name=feature
    )

    if showGridPoints:
        print(pdp_feature.feature_info.grids)

    # Graficamos
    fig, axes = pdp_feature.plot(
        plot_lines=True,
        frac_to_plot=100,
        plot_pts_dist=True,
        plot_params={"pdp_hl": True, "line": {"hl_color": "#f46d43"}}
    )

    if showPlot:
        fig.show()
    else:
        print(pdp_feature.feature_info.grids)
        # fig.write_image(os.path.join(os.path.join(PDP_DIR, dataset_name), fr"pdp_{feature}_{clf_name}.png"))
