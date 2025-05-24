import pandas as pd
from modelos.predecir_con_lgbm import predecir_con_lgbm_anual, predecir_con_lgbm_mensual
from modelos.predecir_con_sarima import predecir_con_sarima
from modelos.predecir_con_winters import predecir_con_winters
import os
import numpy as np

MODELOS_ANUAL_DIR = 'modelos/modelos'
MODELOS_MENSUAL_DIR = 'modelos/modelos_mensuales'

def predecir_ventas(producto, tienda=None, temporada=None):
    ventas_df = pd.read_csv('modelos/ventas.csv', dtype={'Tienda': str})
    modelos_df = pd.read_csv('modelos/mejores_modelos.csv', dtype={'Tienda': str})
    modelos_df_temporada = pd.read_csv('modelos/mejores_modelos_temporada.csv', dtype={'Tienda': str})
    meses = 12

    modelo_info = pd.DataFrame()
    modelo_temporada = pd.DataFrame()
    tapb = None
    modelo_tipo_temporada = None
    if tienda and tienda != 'Todas':
        ventas_filtradas = ventas_df[(ventas_df['Nombre_Producto'] == producto) & (ventas_df['Tienda'] == tienda)]
        modelo_info = modelos_df[(modelos_df['Producto'] == producto) & (modelos_df['Tienda'] == tienda)]
    elif temporada:
        tienda = 'Todas'
        ventas_filtradas = ventas_df[(ventas_df['Nombre_Producto'] == producto) & (ventas_df['Temporada'] == temporada)]
        modelo_temporada = modelos_df_temporada[
            (modelos_df_temporada['Producto'] == producto) & 
            (modelos_df_temporada['Temporada'] == temporada)]
        modelo_tipo_temporada = modelo_temporada.iloc[0]['Modelo_Ganador']
        tapb = modelo_temporada.iloc[0]['TAPB_Min']
    else:
        tienda = 'Todas'
        ventas_filtradas = ventas_df[ventas_df['Nombre_Producto'] == producto]
        modelo_info = modelos_df[(modelos_df['Producto'] == producto) & (modelos_df['Tienda'] == tienda)]

    if ventas_filtradas.empty or (modelo_info.empty and modelo_temporada.empty):
        raise ValueError("No hay datos o modelo para esta combinación")

    temporada = '-' if temporada == None else temporada

    series = ventas_filtradas.groupby('Periodo')['Cantidad'].sum()
    series.index = pd.to_datetime(series.index).to_period('M').to_timestamp()


    modelo_tipo = modelo_info.iloc[0]['Modelo_Ganador'] if not modelo_info.empty else None
    tapb = modelo_info.iloc[0]['TAPB_Min'] if not tapb else tapb
    producto_clean = producto.replace(" ", "_")

    if modelo_tipo == 'LightGBM' or modelo_tipo_temporada == 'LightGBM':
        if temporada == '-':
            modelo_path = os.path.join(MODELOS_ANUAL_DIR, f"modelo_{tienda}_{producto_clean}.pkl")
        else:
            modelo_path = os.path.join(MODELOS_ANUAL_DIR, f"modelo_{tienda}_{temporada}_{producto_clean}.pkl")
        predicciones = predecir_con_lgbm_anual(series, modelo_path)

    elif modelo_tipo == 'SARIMA' or modelo_tipo_temporada == 'SARIMA':
        predicciones  = predecir_con_sarima(series, meses)

    elif modelo_tipo == 'Winters' or modelo_tipo_temporada == 'Winters':
        predicciones = predecir_con_winters(series)
        
    else:
        raise ValueError("Modelo desconocido")
    
    if temporada == '-':
        modelo_mensual_path = os.path.join(MODELOS_MENSUAL_DIR, f"modelo_{tienda}_{producto_clean}.pkl")
    else:
        modelo_mensual_path = os.path.join(MODELOS_MENSUAL_DIR, f"modelo_{tienda}_{temporada}_{producto_clean}.pkl")
    
    predicciones_mensuales = predecir_con_lgbm_mensual(series, modelo_mensual_path)
    

    # métricas
    ventas_anuales = predicciones.sum()
    tendencia = 'sobreestimación' if tapb > 0 else 'subestimación'
    confiabilidad = clasificar_confiabilidad(tapb)
    tapb = float(round(tapb, 2)) if not np.isnan(tapb) else None

    return {
        'producto': producto,
        'tienda': tienda,
        'temporada': temporada,
        'prediccion_anual': predicciones.rename_axis('Periodo').rename(index = lambda x: x.strftime('%Y-%m')).to_dict(),
        'prediccion_anual_total': int(ventas_anuales),
        'tapb': tapb,
        'tendencia_estimacion': tendencia,
        'confiabilidad': confiabilidad,
        'prediccion_mensual': predicciones_mensuales.rename_axis('Periodo').rename(index = lambda x: x.strftime('%Y-%m')).to_dict(),
        'ventas_anteriores': series.rename_axis('Periodo').rename(index = lambda x: x.strftime('%Y-%m')).to_dict()
    }

def clasificar_confiabilidad(tapb):
    valor = abs(tapb)
    if valor < 5:
        return 'muy alta'
    elif valor < 10:
        return 'alta'
    elif valor < 20:
        return 'media'
    else:
        return 'baja'
    
