import optuna
import pandas as pd
import lightgbm as lgb
import numpy as np
import pickle
import os
from sklearn.metrics import mean_absolute_error

# Carpeta para guardar modelos
OUTPUT_DIR = '/mnt/52A3CF4F78FC5DEB/codigo/graficas_RV/predicciones/modelos'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calcular_tab_tapb(y_true, y_pred):
    tab = np.sum(y_pred - y_true)
    tapb = 100 * tab / np.sum(y_true)
    return tab, tapb

def preparar_datos_train_test(window, df, target_col='Cantidad'):
    df = df.copy()
    df['Periodo'] = pd.to_datetime(df['Periodo'])
    df = df.sort_values('Periodo').reset_index(drop=True)

    for i in range(1, window + 1):
        df[f'lag_{i}'] = df[target_col].shift(i)

    df = df.dropna().reset_index(drop=True)

    if df['Periodo'].nunique() < 24:
        print(f'⚠️ ⚠️ ⚠️ ⚠️ ⚠️ ⚠️ Demasiados pocos periodos únicos después de agregar: {df["Periodo"].nunique()}') 
        return None, None, None, None

    max_fecha = df['Periodo'].max()
    corte_fecha = max_fecha - pd.DateOffset(years=1)

    train_df = df[df['Periodo'] <= corte_fecha]
    test_df = df[df['Periodo'] > corte_fecha]

    print(f"[🧪] {len(df)} filas iniciales después de lag window={window}")
    print(f"[🧪] Periodos únicos: {df['Periodo'].nunique()}")

    if len(train_df) < 12 or len(test_df) < 12:
        print(f"[❌] Datos insuficientes para entrenamiento (train={len(train_df)}, test={len(test_df)})")

    feature_cols = [f'lag_{i}' for i in range(1, window + 1)]
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    return X_train, y_train, X_test, y_test

def run_optuna_for_group(df_group, n_trials=30):
    best_model = None
    best_score = float('inf')

    df_group['Periodo'] = pd.to_datetime(df_group['Periodo'])
    df_group = df_group.set_index('Periodo').resample('M').sum().reset_index()

    posibles_windows = [w for w in [3, 6, 12] if df_group.shape[0] - w >= 24]
    if not posibles_windows:
        raise ValueError("❌ Ningún window es viable con los datos disponibles.")

    def objective(trial):
        nonlocal best_model, best_score

        window = trial.suggest_categorical('window', [3, 6, 12])
        num_leaves = trial.suggest_int('num_leaves', 8, 64)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)
        reg_alpha = trial.suggest_float('reg_alpha', 0.0, 2.0)
        reg_lambda = trial.suggest_float('reg_lambda', 0.0, 2.0)

        X_train, y_train, X_test, y_test = preparar_datos_train_test(window, df_group)
        if X_train is None:
            print(f"[⛔] Trial inválido - window={window} → datos insuficientes")
            return float('inf')

        model = lgb.LGBMRegressor(
            objective='huber',
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            verbose=-1,
            seed=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        _, tapb = calcular_tab_tapb(y_test, y_pred)

        score = abs(tapb)
        if score < best_score:
            best_score = score
            best_model = model

        return score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    if best_model is None:
        raise ValueError("No se pudo entrenar ningún modelo válido: todas las particiones fallaron.")

    return study.best_trial.params, study.best_value, best_model

# ------------------------------
# Cargar datos
# ------------------------------
dataframe = pd.read_csv('ventas.csv')

resultados = []

# Definir tiendas y productos
tiendas = [21504,22808,23202,23605,26001,91592,92892,92894,92991,94191]
temporadas = ['OI','PV']
productos = [
    "PANTALON MUJER", 
    "CAMISA MANGA LARGA MUJER", 
    "CAMISETA MUJER", 
    "PUNTO MANGA LARGA MUJER", 
    "CAMISA HOMBRE", 
    "BOLSO BANDOLERA MUJER", 
    "PANTALON HOMBRE", 
]

# ------------------------------
# Entrenar modelos por tienda y producto
# ------------------------------
# for tienda in tiendas:
#     for producto in productos:
#         df_group = dataframe[(dataframe['Tienda'] == tienda) & (dataframe['Nombre_Producto'] == producto)].copy()

#         if df_group.shape[0] < 30:
#             print(f'Saltando tienda {tienda}, producto {producto}, pocos datos.')
#             continue

#         print(f'Optimizando tienda {tienda}, producto {producto}...')
#         try:
#             best_params, best_tapb, model = run_optuna_for_group(df_group)
#             resultados.append({
#                 'Tienda': tienda,
#                 'Producto': producto,
#                 'Modelo': 'LightGBM',
#                 'TAPB': best_tapb,
#                 **best_params
#             })

#             # Guardar modelo
#             filename = f"modelo_{tienda}_{producto}.pkl".replace(" ", "_")
#             filepath = os.path.join(OUTPUT_DIR, filename)
#             with open(filepath, 'wb') as f:
#                 pickle.dump({'model': model, 'window': best_params['window']}, f)

#         except Exception as e:
#             print(f'Error en tienda {tienda}, producto {producto}: {e}')

# ------------------------------
# Entrenar modelos por temporada y producto
# ------------------------------
for temporada in temporadas:
    for producto in productos:
        df_group = dataframe[(dataframe['Temporada'] == temporada) & (dataframe['Nombre_Producto'] == producto)].copy()
        df_group['Periodo'] = pd.to_datetime(df_group['Periodo'])

        df_group = df_group.groupby('Periodo').agg({'Cantidad': 'sum'}).reset_index()


        if 'Cantidad' not in df_group.columns:
            df_group['Cantidad'] = 0

        if df_group.shape[0] < 30 or df_group['Periodo'].nunique() < 24: 
            print(f'​Saltando temporada {temporada}, producto {producto}, pocos datos o periodos.')
            continue

        print(f'Optimizando temporada {temporada}, producto {producto}...')
        try:
            best_params, best_tapb, model = run_optuna_for_group(df_group)
            resultados.append({
                'Tienda': 'Todas',
                'Temporada': temporada,
                'Producto': producto,
                'Modelo': 'LightGBM',
                'TAPB': best_tapb,
                **best_params
            })
            print("Model entrenado:", model)
            filename = f"modelo_Todas_{temporada}_{producto}.pkl".replace(" ", "_")
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, 'wb') as f:
                pickle.dump({'model': model, 'window': best_params['window']}, f)

        except Exception as e:
            print(f'Error en temporada {temporada}, producto {producto}: {e}')

# ------------------------------
# Entrenar modelos por producto (todas las tiendas juntas)
# ------------------------------
# for producto in productos:
#     df_group = dataframe[dataframe['Nombre_Producto'] == producto].copy()
#     df_group['Periodo'] = pd.to_datetime(df_group['Periodo'])

#     df_group = df_group.groupby('Periodo').agg({'Cantidad': 'sum'}).reset_index()

#     if df_group.shape[0] < 30 or df_group['Periodo'].nunique() < 24:  # <-- CAMBIO
#         print(f'Saltando producto {producto} (global), pocos datos o periodos.')
#         continue

#     print(f'Optimizando producto {producto} (global)...')
#     try:
#         best_params, best_tapb, model = run_optuna_for_group(df_group)
#         resultados.append({
#             'Tienda': 'Todas',
#             'Producto': producto,
#             'Modelo': 'LightGBM',
#             'TAPB': best_tapb,
#             **best_params
#         })

#         filename = f"modelo_Todas_{producto}.pkl".replace(" ", "_")
#         filepath = os.path.join(OUTPUT_DIR, filename)
#         with open(filepath, 'wb') as f:
#             pickle.dump({'model': model, 'window': best_params['window']}, f)

#     except Exception as e:
#         print(f'Error en producto {producto} (global): {e}')

# ------------------------------
# Guardar resultados finales
# ------------------------------
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('mejores_parametros_por_tienda_producto.csv', index=False)
