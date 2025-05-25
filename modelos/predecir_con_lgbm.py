import pandas as pd
import numpy as np
import pickle
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters

def predecir_con_lgbm_anual(series, modelo_path):
    with open(modelo_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        window = data['window']
        

    series = series.sort_index()
    serie_pred = series.copy()
    predicciones = []
    fechas_pred = []

    print(data)

    for i in range(12):
        # Extraer los últimos 'window' valores como lags
        lags = [serie_pred.iloc[-j] for j in range(window, 0, -1)]
        X_input = pd.DataFrame([lags], columns=[f'lag_{i}' for i in range(1, window + 1)])

        # Predecir siguiente valor
        pred = model.predict(X_input)[0]
        predicciones.append(pred)

        # Crear la nueva fecha para la predicción
        last_date = serie_pred.index[-1]
        next_month = (last_date + pd.DateOffset(months=1)).replace(day=1)
        fechas_pred.append(next_month)

        # Agregar predicción a la serie para usarla como lag en la próxima iteración
        serie_pred.loc[next_month] = pred

    pred_series = pd.Series(predicciones, index=fechas_pred)

    return pred_series

def predecir_con_lgbm_mensual(serie, modelo_path, pasos=12):
    with open(modelo_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        window = data['window']

    serie = serie.sort_index()
    serie_pred = serie.copy()

    predicciones = []
    fechas_pred = []

    for i in range(pasos):
        # Preparar ventana rolling para tsfresh
        df_tsfresh = serie_pred.reset_index()
        df_tsfresh.columns = ['Periodo', 'Cantidad']

        rows = []
        if len(df_tsfresh) < window:
            # No hay datos suficientes para crear features, devolver NaNs
            predicciones.extend([np.nan]*(pasos - i))
            break

        # Seleccionamos la última ventana para extraer features
        window_df = df_tsfresh.iloc[-window:].copy()
        window_df['id'] = 0
        df_rolling = window_df

        features = extract_features(
            df_rolling[['id', 'Periodo', 'Cantidad']],
            column_id='id',
            column_sort='Periodo',
            default_fc_parameters=EfficientFCParameters(),
            disable_progressbar=True
        )
        impute(features)

        # Crear features manuales
        serie_temp = serie_pred.to_frame(name='Cantidad')
        serie_temp.index = pd.to_datetime(serie_temp.index)
        serie_temp = serie_temp.asfreq('MS').fillna(0)
        df_manual = create_manual_features(serie_temp)
        df_manual = df_manual.drop(columns=['Cantidad'])
        df_manual = df_manual.iloc[-1:]  # solo la última fila

        # Seleccionar features comunes entre train y test
        X = pd.concat([features.reset_index(drop=True), df_manual.reset_index(drop=True)], axis=1)
        X = X.select_dtypes(include=[np.number])
        X = clean_feature_names(X)

        features_entrenamiento = data['features']  
        X = X.reindex(columns=features_entrenamiento, fill_value=0)

        # Predecir
        pred = model.predict(X)[0]
        predicciones.append(int(pred))

        # Añadir predicción a la serie para siguiente iteración
        last_date = serie_pred.index[-1]
        next_date = (last_date + pd.DateOffset(months=1)).replace(day=1)
        fechas_pred.append(next_date)
        serie_pred.loc[next_date] = pred

        print(data)

    pred_series = pd.Series(predicciones, index=fechas_pred)
    return pred_series


def create_manual_features(df):
    df = df.copy()
    df['lag_1']  = df['Cantidad'].shift(1)
    df['lag_3']  = df['Cantidad'].shift(3)
    df['lag_12'] = df['Cantidad'].shift(12)
    df['r12_1']  = (df['Cantidad'] - df['lag_12']) / (df['lag_12'] + 1e-6)
    df['month']  = df.index.month
    df['quarter']= df.index.quarter
    return df

def clean_feature_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace('[^A-Za-z0-9_]+', '_', regex=True)
        .str.strip('_')
    )
    return df

