import pandas as pd
from modelos import predecir_con_lgbm
import os
import numpy as np

MODELOS_ANUAL_DIR = 'modelos/modelos'
MODELOS_MENSUAL_DIR = 'modelos/modelos_mensuales'

def predecir_ventas(producto, tienda=None):
    ventas_df = pd.read_csv('modelos/ventas.csv', dtype={'Tienda': str})
    modelos_df = pd.read_csv('modelos/mejores_modelos.csv', dtype={'Tienda': str})

    if tienda:
        ventas_filtradas = ventas_df[(ventas_df['Nombre_Producto'] == producto) & (ventas_df['Tienda'] == tienda)]
        modelo_info = modelos_df[(modelos_df['Producto'] == producto) & (modelos_df['Tienda'] == tienda)]
    else:
        ventas_filtradas = ventas_df[ventas_df['Nombre_Producto'] == producto]
        modelo_info = modelos_df[(modelos_df['Producto'] == producto) & (modelos_df['Tienda'] == 'Todas')]

    if ventas_filtradas.empty or modelo_info.empty:
        raise ValueError("No hay datos o modelo para esta combinación")

    series = ventas_filtradas.groupby('Periodo')['Cantidad'].sum()
    series.index = pd.to_datetime(series.index).to_period('M').to_timestamp()

    modelo_tipo = modelo_info.iloc[0]['Modelo_Ganador']
    producto_clean = producto.replace(" ", "_")

    if modelo_tipo == 'LightGBM':
        modelo_path = os.path.join(MODELOS_ANUAL_DIR, f"modelo_{tienda}_{producto_clean}.pkl")
        predicciones, tapb = predecir_con_lgbm.predecir_con_lgbm_anual(series, modelo_path)

    elif modelo_tipo == 'SARIMA':
        # predicciones, tapb = predecir_con_sarima(series)
        # ventas_anuales = predicciones.sum()
        print(modelo_tipo)
    
    elif modelo_tipo == 'Winters':
        # predicciones, tapb = predecir_con_winters(series)
        # ventas_anuales = predicciones.sum()
        print(modelo_tipo)
    else:
        raise ValueError("Modelo desconocido")
    
    modelo_mensual_path = os.path.join(MODELOS_MENSUAL_DIR, f"modelo_{tienda}_{producto_clean}.pkl")
    predicciones_mensuales = predecir_con_lgbm.predecir_con_lgbm_mensual(series, modelo_mensual_path)

    # métricas
    ventas_anuales = predicciones.sum()
    tendencia = 'sobreestimación' if tapb > 0 else 'subestimación'
    confiabilidad = clasificar_confiabilidad(tapb)
    tapb = float(round(tapb, 2)) if not np.isnan(tapb) else None

    # # graficas
    # fig_path = f'graficas/{producto}_{tienda or "total"}.png'
    # generar_grafico(series, predicciones, fig_path)

    return {
        'producto': producto,
        'tienda': tienda,
        'prediccion_anual_total': int(ventas_anuales),
        'tapb': tapb,
        'tendencia_estimacion': tendencia,
        'confiabilidad': confiabilidad,
        'prediccion_mensual': predicciones_mensuales.rename_axis('Periodo').rename(index=lambda x: x.strftime('%Y-%m')).to_dict(),
        'ventas_anteriores': series.rename_axis('Periodo').rename(index=lambda x: x.strftime('%Y-%m')).to_dict()
        # 'grafico': fig_path
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
    
