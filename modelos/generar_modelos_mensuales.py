import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Configuración ---
INPUT_CSV = 'ventas.csv'
OUTPUT_DIR = '/mnt/52A3CF4F78FC5DEB/codigo/graficas_RV/predicciones/modelos_mensuales'
os.makedirs(OUTPUT_DIR, exist_ok=True)

window = 12
test_size = 12

params = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'seed': 42,
    'verbose': -1
}

tiendas = [21504, 22808, 23202, 23605, 26001, 91592, 92892, 92894, 92991, 94191]
familias_top = [
    "PANTALON MUJER", 
    "CAMISA MANGA LARGA MUJER", 
    "CAMISETA MUJER", 
    "PUNTO MANGA LARGA MUJER", 
    "CAMISA HOMBRE", 
    "BOLSO BANDOLERA MUJER", 
    "PANTALON HOMBRE", 
]

# --- Funciones auxiliares ---

def create_manual_features(df):
    df = df.copy()
    # lags
    df['lag_1'] = df['Cantidad'].shift(1)
    df['lag_3'] = df['Cantidad'].shift(3)
    df['lag_6'] = df['Cantidad'].shift(6)
    df['lag_12'] = df['Cantidad'].shift(12)

    # ratios 
    df['r12_1'] = (df['Cantidad'] - df['lag_12']) / (df['lag_12'] + 1e-6)

    # rstacionalidad 
    month = df.index.month
    df['sin_month'] = np.sin(2 * np.pi * month / 12)
    df['cos_month'] = np.cos(2 * np.pi * month / 12)

    # mes y trimestre (numéricos)
    df['month'] = month
    df['quarter'] = df.index.quarter
    return df

def clean_feature_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace('[^A-Za-z0-9_]+', '_', regex=True)
        .str.strip('_')
    )
    return df

def entrenar_modelo(grupo, tienda, producto, temporada=None):
    grupo = grupo.groupby('Periodo', as_index=False)['Cantidad'].sum()
    if grupo.empty:
        return

    # establecer frecuencia mensual
    grupo = grupo.set_index('Periodo').asfreq('MS')
    grupo['Cantidad'] = grupo['Cantidad'].fillna(0)

    if (grupo['Cantidad'] > 0).sum() < 24:
        return

    # target: siguiente mes
    grupo['target'] = grupo['Cantidad'].shift(-1)
    grupo = grupo.dropna(subset=['target'])

    df_feat = create_manual_features(grupo)
    df_feat = df_feat.drop(columns=['Cantidad', 'target']).iloc[window:]
    y = grupo['target'].iloc[window:]

    X = df_feat.select_dtypes(include=[np.number])
    X = clean_feature_names(X)

    if len(X) <= test_size:
        return

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(10)
        ]
    )

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Modelo {producto} | Tienda {tienda} | Temporada {temporada or 'N/A'} => MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    producto_safe = producto.replace('/', '_').replace(' ', '_')
    if temporada:
        model_filename = f"modelo_Todas_{temporada}_{producto_safe}.pkl"
    elif tienda == 'Todas':
        model_filename = f"modelo_Todas_{producto_safe}.pkl"
    else:
        model_filename = f"modelo_{tienda}_{producto_safe}.pkl"

    model_path = os.path.join(OUTPUT_DIR, model_filename)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'window': window,
            'features': list(X.columns)
        }, f)

# --- Carga y preparación de datos ---

ventas = pd.read_csv(INPUT_CSV, parse_dates=['Periodo'])
ventas['Periodo'] = ventas['Periodo'].dt.to_period('M').dt.to_timestamp()

ventas = ventas[
    (ventas['Tienda'].isin(tiendas)) & 
    (ventas['Nombre_Producto'].isin(familias_top))
]

ventas = ventas.groupby(['Periodo', 'Tienda', 'Nombre_Producto', 'Temporada'])['Cantidad'].sum().reset_index()

# --- Entrenamiento ---

# modelos por tienda y producto
for (tienda, producto), grupo in tqdm(ventas.groupby(['Tienda', 'Nombre_Producto']), desc='Modelos por tienda-producto'):
    entrenar_modelo(grupo, tienda, producto)

# modelos globales por producto (todas las tiendas)
for producto in tqdm(familias_top, desc='Modelos globales por producto'):
    grupo_producto = ventas[ventas['Nombre_Producto'] == producto].copy()
    grupo_producto = grupo_producto.groupby('Periodo')['Cantidad'].sum().reset_index()
    grupo_producto['Tienda'] = 'Todas'
    grupo_producto['Nombre_Producto'] = producto
    entrenar_modelo(grupo_producto, 'Todas', producto)

# modelos por temporada y producto (global)
for (temporada, producto), grupo in tqdm(ventas.groupby(['Temporada', 'Nombre_Producto']), desc='Modelos por temporada-producto'):
    grupo = grupo.groupby('Periodo')['Cantidad'].sum().reset_index()
    grupo['Tienda'] = 'Todas'
    grupo['Nombre_Producto'] = producto
    grupo['Temporada'] = temporada
    entrenar_modelo(grupo, 'Todas', producto, temporada=temporada)
