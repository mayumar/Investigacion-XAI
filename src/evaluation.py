# src/evaluation.py

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

def evaluar_modelo(metricas, clf_name, y_test, prediction, normalizado, contaminacion, duracion, semilla):
    # Calculamos las metricas para test
    cm = confusion_matrix(y_test, prediction)
    print('Matriz de confusión: \n', cm)

    # Métricas personalizadas desde matriz de confusión
    sensibilidad = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    especificidad = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0

    # Métricas generales
    roc_auc = roc_auc_score(y_test, prediction)
    acc = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    # Añadir resultados al DataFrame
    nueva_fila = {
        'Modelo': clf_name,
        'Semilla': semilla,
        'Normalizado': normalizado,
        'Contaminacion': contaminacion,
        'TN' : cm[0,0],
        'FP' : cm[0,1],
        'FN' : cm[1,0],
        'TP' : cm[1,1],
        'Accuracy': acc,
        'F1-score': f1,
        'Sensibilidad': sensibilidad,
        'Especificidad': especificidad,
        'Precisión': precision,
        'ROC-AUC': roc_auc,
        'Tiempo (s)': duracion
    }
    nueva_df = pd.DataFrame([nueva_fila])

    # Si metricas está vacío, lo reemplazamos; si no, concatenamos
    if metricas.empty:
        metricas = nueva_df
    else:
        metricas = pd.concat([metricas, nueva_df], ignore_index=True)


    # Lista de intervalos: cada tupla es (inicio, fin)
    intervalos_evaluacion = [
        # (pd.Timestamp("2020-04-12 11:50:00"), pd.Timestamp("2020-04-12 23:30:00")),# Train
        # (pd.Timestamp("2020-04-18 00:00:00"), pd.Timestamp("2020-04-19 01:30:00")),#
        # (pd.Timestamp("2020-04-29 03:20:00"), pd.Timestamp("2020-04-29 04:00:00")),
        # (pd.Timestamp("2020-04-29 22:00:00"), pd.Timestamp("2020-04-29 22:20:00")),
        # (pd.Timestamp("2020-05-13 14:00:00"), pd.Timestamp("2020-05-13 23:59:00")),
        # (pd.Timestamp("2020-05-18 05:00:00"), pd.Timestamp("2020-05-18 05:30:00")),
        # (pd.Timestamp("2020-05-19 10:10:00"), pd.Timestamp("2020-05-19 11:00:00")),
        # (pd.Timestamp("2020-05-19 22:10:00"), pd.Timestamp("2020-05-20 20:00:00")),
        # (pd.Timestamp("2020-05-23 09:50:00"), pd.Timestamp("2020-05-23 10:10:00")),
        # (pd.Timestamp("2020-05-29 23:30:00"), pd.Timestamp("2020-05-30 06:00:00")),#
        # (pd.Timestamp("2020-06-01 15:00:00"), pd.Timestamp("2020-06-01 15:40:00")),
        # (pd.Timestamp("2020-06-03 10:00:00"), pd.Timestamp("2020-06-03 11:00:00")),
        # (pd.Timestamp("2020-06-05 10:00:00"), pd.Timestamp("2020-06-07 14:30:00")),#
        # (pd.Timestamp("2020-07-08 17:30:00"), pd.Timestamp("2020-07-08 19:00:00")),# <------ (1h30) Test empieza aquí
        # (pd.Timestamp("2020-07-15 14:30:00"), pd.Timestamp("2020-07-15 19:00:00")),# <------ (4h30)
        # (pd.Timestamp("2020-07-17 04:30:00"), pd.Timestamp("2020-07-17 05:30:00"))
    ]

    # print('*** Fallos ***')
    # for inicio, fin in intervalos_evaluacion:
    #     # Crear el DataFrame combinando los valores reales y predichos
    #     df = pd.DataFrame({
    #         'Real': y_test,
    #         'Predicho': prediction
    #     }, index=y_test.index)

    #     # Filtrar errores
    #     df_errores_inicial = df[df['Real'] != df['Predicho']]

    #     # Filtrar por rango de fechas
    #     df_errores = df_errores_inicial.loc[inicio:fin]

    #     if not df_errores.empty:
    #         print(f"Evento: {inicio} / {fin}")

    #         print(df_errores.to_string())

    #         # Graficar los errores
    #         plt.figure(figsize=(12, 5))
    #         plt.scatter(df_errores.index, df_errores['Real'], color='blue', alpha=0.6, label='Real', marker='o')
    #         plt.scatter(df_errores.index, df_errores['Predicho'], color='red', alpha=0.4, label='Predicho', marker='x')

    #         plt.title(f'Errores de Clasificación - {clf_name}')
    #         plt.xlabel('Fecha')
    #         plt.ylabel('Clase')
    #         plt.yticks([0, 1])
    #         plt.legend()
    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.xticks(rotation=45)

    #         plt.xlim(pd.to_datetime(inicio), pd.to_datetime(fin))


    #         plt.show()
        
    return metricas
