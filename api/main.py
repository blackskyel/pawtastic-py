import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os

# Ruta relativa desde la raíz del proyecto
MODEL_FILE = "model/pet_adoption_multimodel.pkl"

# --- 1. CARGAR EL MODELO ENTRENADO ---
try:
    with open(MODEL_FILE, 'rb') as f:
        artifacts = pickle.load(f)
    
    regressor = artifacts['regressor']
    classifiers = artifacts['classifiers']
    top_breeds = artifacts['top_breeds']
    top_colors = artifacts['top_colors']
    model_columns = artifacts['model_columns']
    mae_error = artifacts['mae']
    
    print("✅ Modelos cargados correctamente.")
except FileNotFoundError:
    raise RuntimeError(f"❌ No se encontró el archivo {MODEL_FILE}. Ejecuta train_multimodel_final.py primero.")

# --- 2. DEFINIR EL ESQUEMA DE DATOS (VALIDACIÓN) ---
# Esto asegura que quien use la API envíe los datos correctos
class PetInput(BaseModel):
    animaltype: int  # 0 o 1 (según tu dataset)
    gender: int      # 0 o 1
    petsize: int     # 0 a 4
    breed: str       # Texto, ej: "Labrador"
    color: str       # Texto, ej: "Black"

class PredictionOutput(BaseModel):
    dias_estimados: int
    rango_estimado: str
    confianza_modelo: str
    probabilidades_temporales: dict

app = FastAPI(title="Pawtastic Prediction API", version="1.0")

# --- 3. LÓGICA DE PREDICCIÓN ---
def prepare_features(input_data: PetInput):
    """Preprocesa los datos IGUAL que en el entrenamiento"""
    
    # 1. Crear DataFrame de una sola fila
    data = {
        'animaltype': [input_data.animaltype],
        'gender': [input_data.gender],
        'petsize': [input_data.petsize],
        'breed': [input_data.breed.strip().upper()], # Misma limpieza que train
        'color': [input_data.color.strip().upper()]
    }
    df = pd.DataFrame(data)
    
    # 2. Aplicar Top N (Si no está en la lista, es 'Other')
    # OJO: Usamos las listas que cargamos del pickle, no recalculamos nada
    df['breed'] = np.where(df['breed'].isin(top_breeds), df['breed'], 'Other')
    df['color'] = np.where(df['color'].isin(top_colors), df['color'], 'Other')
    
    # 3. One-Hot Encoding
    df = pd.get_dummies(df, columns=['breed', 'color'])
    
    # 4. ALINEACIÓN DE COLUMNAS (CRÍTICO EN PRODUCCIÓN)
    # El modelo espera exactamente las mismas columnas que en el train.
    # Al hacer get_dummies con 1 sola fila, faltarán muchas columnas.
    # 'reindex' agrega las columnas faltantes con valor 0.
    df = df.reindex(columns=model_columns, fill_value=0)
    
    return df

@app.post("/predict", response_model=PredictionOutput)
def predict_adoption(pet: PetInput):
    
    # A. Preprocesar
    try:
        processed_df = prepare_features(pet)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error procesando datos: {str(e)}")

    # B. Predicción Regresor (Días exactos)
    dias_pred = regressor.predict(processed_df)[0]
    dias_pred = int(max(1, dias_pred)) # Evitar negativos
    
    # C. Predicción Clasificadores (Probabilidades)
    # Lógica inteligente: Mostrar probabilidades cercanas a la predicción
    probs_response = {}
    
    # Definimos qué hitos mostrar según la predicción de días
    hitos = [30, 60, 90, 120, 150]
    hitos_relevantes = []
    
    # Si predice 40 días, nos interesa saber la prob de <60 y <90
    for hito in hitos:
        if dias_pred < hito:
            hitos_relevantes.append(hito)
            # Agregamos el siguiente para contexto (si existe)
            idx = hitos.index(hito)
            if idx + 1 < len(hitos):
                hitos_relevantes.append(hitos[idx+1])
            break
            
    # Si la predicción es muy alta (>150), mostramos el último hito
    if not hitos_relevantes:
        hitos_relevantes = [150]

    # Calculamos prob para esos hitos
    for h in hitos_relevantes:
        # predict_proba devuelve [[prob_no, prob_si]]
        prob = classifiers[h].predict_proba(processed_df)[0][1]
        probs_response[f"adopcion_en_menos_de_{h}_dias"] = f"{prob*100:.1f}%"

    # D. Respuesta
    return {
        "dias_estimados": dias_pred,
        "rango_estimado": f"Entre {max(1, dias_pred - int(mae_error))} y {dias_pred + int(mae_error)} días",
        "confianza_modelo": f"Margen de error promedio: +/- {mae_error:.1f} días",
        "probabilidades_temporales": probs_response
    }

# Endpoint de salud para verificar que la API corre
@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": True}