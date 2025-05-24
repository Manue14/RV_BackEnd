import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def predecir_con_winters(series, pasos=12, estacionalidad=12):
    series = series.sort_index()

    if series.empty or series.sum() == 0:
        print(f"Skipping prediccion de Winters porque no tiene datos válidos")
        return None

    try:
        modelo = ExponentialSmoothing(
            series,
            trend='add',
            seasonal='add',
            seasonal_periods=estacionalidad
        ).fit()

        pred = modelo.forecast(pasos)

    except Exception as e:
        print(f'Error al predecir con Winters: {e}')
        return None
    

    # Crear fechas futuras
    ultima_fecha_serie = series.index[-1]
    fechas = pd.date_range(
        start=(ultima_fecha_serie + pd.DateOffset(months=1)).replace(day=1),
        periods=pasos,
        freq='MS'
    )

    pred.index = fechas
    return pred