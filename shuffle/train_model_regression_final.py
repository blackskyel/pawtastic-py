import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import datetime # Necesario para el log

# --- CONFIGURACIÓN ---
CSV_PATH = '../dataset/pet_adoption_dataset.csv'
MODEL_FILENAME = '../model/pet_adopcion_model_regression_optimizado.pkl'
LOG_FILENAME = '../model/training_log_regression_optimizado.txt' # Archivo donde se guardará el historial

def load_and_clean_data(filepath):
    print("1. Limpieza profunda de datos...")
    # Leemos todo como texto para evitar errores de mezcla de tipos
    df = pd.read_csv(filepath, dtype=str)
    
    # Convertir columnas numéricas
    cols_num = ['animaltype', 'gender', 'petsize', 'duration']
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- LIMPIEZA DE VALORES IMPOSIBLES ---
    df = df.dropna(subset=['duration'])
    df = df[df['duration'] >= 0] # Sin tiempos negativos
    
    # Eliminar el pico artificial de 300 si consideras que es error (opcional)
    df = df[df['duration'] != 300] 

    # Eliminar Outliers Extremos (Percentil 99)
    q99 = df['duration'].quantile(0.99)
    df = df[df['duration'] <= q99]
    
    # Rellenar nulos restantes con moda
    df = df.fillna(df.mode().iloc[0])

    # Limpieza de texto
    df['breed'] = df['breed'].astype(str).str.strip().str.upper()
    df['color'] = df['color'].astype(str).str.strip().str.upper()
    
    # Normalizar 'NAN' string a UNKNOWN
    df['breed'] = df['breed'].replace(['NAN', 'NAN.', ''], 'UNKNOWN')
    df['color'] = df['color'].replace(['NAN', 'NAN.', ''], 'UNKNOWN')
    
    print(f"   -> Registros válidos para entrenar: {len(df)}")
    return df

def save_log(mae, r2, top_breeds, top_colors):
    """Guarda los resultados y configuración en un archivo de texto"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_content = f"""
================================================
REPORTE MODELO OPTIMIZADO (GRADIENT BOOSTING)
Fecha: {timestamp}
================================================
MÉTRICAS:
- MAE (Error Promedio): +/- {mae:.2f} días
- R2 Score: {r2:.4f}

CONFIGURACIÓN:
- Cantidad de Razas Top: {len(top_breeds)}
- Lista Razas: {top_breeds}

- Cantidad de Colores Top: {len(top_colors)}
- Lista Colores: {top_colors}
================================================
"""
    # 'a' significa append (agregar al final sin borrar lo anterior)
    with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
        f.write(log_content)
    print(f"--> Log guardado exitosamente en: {LOG_FILENAME}")

def process_and_train(df):
    # Preprocesamiento
    X = df.drop(columns=['duration', 'ID'])
    y = df['duration']
    
    # Calcular Top N
    top_breeds = X['breed'].value_counts().nlargest(12).index.tolist()
    top_colors = X['color'].value_counts().nlargest(10).index.tolist()
    
    # Aplicar agrupación
    X['breed'] = np.where(X['breed'].isin(top_breeds), X['breed'], 'Other')
    X['color'] = np.where(X['color'].isin(top_colors), X['color'], 'Other')
    
    # One-Hot Encoding
    X = pd.get_dummies(X, columns=['breed', 'color'], dummy_na=False)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- ENTRENAMIENTO ---
    print("2. Entrenando Gradient Boosting (optimizando MAE)...")
    model = GradientBoostingRegressor(
        n_estimators=200,       
        learning_rate=0.05,     
        max_depth=5,
        random_state=42,
        loss='absolute_error' # Enfocado en minimizar el error absoluto (días)
    )
    model.fit(X_train, y_train)
    
    # Evaluación
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"\n--- RESULTADO FINAL ---")
    print(f"Nuevo MAE: {mae:.2f} días")
    
    # --- GUARDADO DE LOG Y MODELO ---
    
    # 1. Guardar Log Humano
    save_log(mae, r2, top_breeds, top_colors)
    
    # 2. Guardar Pickle para la API
    artifacts = {
        'model': model,
        'top_breeds': top_breeds,
        'top_colors': top_colors,
        'model_columns': X.columns.tolist(),
        'mae': mae
    }
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"--> Modelo binario (.pkl) guardado en: {MODEL_FILENAME}")

if __name__ == "__main__":
    df = load_and_clean_data(CSV_PATH)
    process_and_train(df)