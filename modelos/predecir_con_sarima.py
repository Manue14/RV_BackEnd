import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.ticker import MultipleLocator
import os
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


def predecir_con_sarima(series, pasos=12):
    series = series.sort_index()

    if series.empty or series.sum() == 0:
        print(f"Skipping prediccion de SARIMA porque no tiene datos válidos")
        return None

    try:
        modelo_auto = auto_arima(series, seasonal=True, m=12, suppress_warnings=True, error_action="ignore")
        orden = modelo_auto.order
        orden_estacional = modelo_auto.seasonal_order
    except Exception as e:
        print(f"ERROR EN EL MODELO AUTO_ARIMA" , e)
        return None
    
    try:
        modelo = SARIMAX(series, order=orden, seasonal_order=orden_estacional,
                         enforce_stationarity=False, enforce_invertibility=False)
        modelo_fit = modelo.fit(disp=False)
    except Exception as e:
        print(f"ERROR EN EL MODELO SARIMA" , e)
        return None
    
    # Crear fechas futuras
    ultima_fecha_serie = series.index[-1]
    fechas = pd.date_range(
        start=(ultima_fecha_serie + pd.DateOffset(months=1)).replace(day=1),
        periods=pasos,
        freq='MS'
    )

    # Hacer predicción
    pred = modelo_fit.forecast(steps=pasos)
    pred.index = fechas

    return pred


