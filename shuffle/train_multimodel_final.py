import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
import pickle
import datetime

# --- CONFIGURACIÓN ---
CSV_PATH = '../dataset/pet_adoption_dataset.csv'
MODEL_FILENAME = '../model/pet_adoption_multimodel.pkl'
LOG_FILENAME = '../model/training_log_multimodel2.txt'
TIME_THRESHOLDS = [30, 60, 90, 120, 150]

def load_and_clean_data(filepath):
    print("1. Cargando y aplicando 'Higiene Agresiva' a los datos...")
    df = pd.read_csv(filepath, dtype=str)
    
    cols_num = ['animaltype', 'gender', 'petsize', 'duration']
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 1. Eliminar nulos y negativos
    df = df.dropna(subset=['duration'])
    df = df[df['duration'] >= 0]
    
    # 2. ELIMINAR EL PICO ARTIFICIAL DE 300 DÍAS
    count_300 = df[df['duration'] == 300].shape[0]
    if count_300 > 0:
        print(f"   -> Eliminando {count_300} registros sospechosos de exactamente 300 días...")
        df = df[df['duration'] != 300]

    # 3. Eliminar Outliers (Q95)
    q95 = df['duration'].quantile(0.95)
    print(f"   -> Cortando outliers superiores a {q95} días")
    df = df[df['duration'] <= q95]
    
    # Rellenar nulos
    df = df.fillna(df.mode().iloc[0])
    df['breed'] = df['breed'].astype(str).str.strip().str.upper().replace(['NAN', ''], 'UNKNOWN')
    df['color'] = df['color'].astype(str).str.strip().str.upper().replace(['NAN', ''], 'UNKNOWN')
    
    print(f"   -> Registros finales para entrenar: {len(df)}")
    return df

def get_processed_features(df):
    X = df.drop(columns=['duration', 'ID'])
    y_reg = df['duration']
    
    top_breeds = X['breed'].value_counts().nlargest(12).index.tolist()
    top_colors = X['color'].value_counts().nlargest(10).index.tolist()
    
    X['breed'] = np.where(X['breed'].isin(top_breeds), X['breed'], 'Other')
    X['color'] = np.where(X['color'].isin(top_colors), X['color'], 'Other')
    
    X = pd.get_dummies(X, columns=['breed', 'color'], dummy_na=False)
    return X, y_reg, top_breeds, top_colors

def save_log(mae, classifier_metrics, top_breeds, top_colors):
    """Guarda métricas y las listas de variables en el log"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metrics_str = "\n".join([f"   - Probabilidad < {d} días: Accuracy {acc:.1%}" for d, acc in classifier_metrics.items()])
    
    # Formateamos las listas para que se lean bien
    breeds_str = ", ".join(top_breeds)
    colors_str = ", ".join(top_colors)
    
    log_content = f"""
================================================
REPORTE FINAL (LIMPIEZA AGRESIVA) - {timestamp}
================================================
1. REGRESOR (DÍAS EXACTOS)
   - MAE (Error Promedio): +/- {mae:.2f} días

2. CLASIFICADORES (PROBABILIDADES)
{metrics_str}

3. VARIABLES TOP SELECCIONADAS (DICCIONARIO)
   - Razas Top ({len(top_breeds)}): 
     [{breeds_str}]
     
   - Colores Top ({len(top_colors)}): 
     [{colors_str}]
================================================
"""
    with open(LOG_FILENAME, 'a', encoding='utf-8') as f:
        f.write(log_content)
    print(f"--> Log actualizado en {LOG_FILENAME}")

def main():
    # 1. Preparación
    df = load_and_clean_data(CSV_PATH)
    X, y_reg, top_breeds, top_colors = get_processed_features(df)
    model_columns = X.columns.tolist()
    
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    # 2. Regresor
    print(f"2. Entrenando Regresor Principal...")
    regressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, loss='absolute_error', random_state=42)
    regressor.fit(X_train, y_reg_train)
    
    preds = regressor.predict(X_test)
    mae = mean_absolute_error(y_reg_test, preds)
    print(f"   -> MAE LOGRADO: {mae:.2f} días")
    
    # 3. Clasificadores
    classifiers = {}
    classifier_metrics = {}
    print("3. Entrenando Clasificadores de Probabilidad...")
    
    for days in TIME_THRESHOLDS:
        y_class_train = (y_reg_train <= days).astype(int)
        y_class_test = (y_reg_test <= days).astype(int)
        
        clf = LogisticRegression(max_iter=1000, solver='liblinear')
        clf.fit(X_train, y_class_train)
        
        acc = accuracy_score(y_class_test, clf.predict(X_test))
        classifiers[days] = clf
        classifier_metrics[days] = acc

    # 4. Guardar
    artifacts = {
        'regressor': regressor,
        'classifiers': classifiers,
        'top_breeds': top_breeds,
        'top_colors': top_colors,
        'model_columns': model_columns,
        'mae': mae
    }
    
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(artifacts, f)
        
    # AQUÍ EL CAMBIO: Pasamos las listas a la función
    save_log(mae, classifier_metrics, top_breeds, top_colors)
    print(f"\n¡Listo! Sistema guardado en '{MODEL_FILENAME}' y reporte en '{LOG_FILENAME}'")

if __name__ == "__main__":
    main()