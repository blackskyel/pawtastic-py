import pandas as pd
import numpy as np

# 1. Cargar datos
df = pd.read_csv('../dataset/pet_adoption_dataset.csv')

# --- A. Mapeo Manual (Ordinal y Binario) ---
# Definimos el orden explícitamente para el tamaño
# mapa_tamanio = {'Small': 1, 'Medium': 2, 'Large': 3, 'Extra Large': 4}
# df['Size_Num'] = df['Size'].map(mapa_tamanio)

# Binario para Vacunado
# df['Vaccinated_Num'] = df['Vaccinated'].map({'Yes': 1, 'No': 0})

# --- B. One-Hot Encoding (Nominal) ---
# Convertimos Color y PetType. 
# 'prefix' ayuda a identificar las columnas después (ej. Color_Black)
# df = pd.get_dummies(df, columns=['Color', 'PetType'], prefix=['Color', 'Type'])

# --- C. Resultado Final ---
# Ahora borramos las columnas de texto originales que ya no sirven
# df = df.drop(columns=['Size', 'Vaccinated'])

print(df.head())






# 1. Definir cuántas razas queremos mantener (ej. las 10 más comunes)
TOP_N = 12

# 2. Calcular cuáles son esas razas top
# .value_counts() cuenta las frecuencias
# .nlargest(n) se queda con las n mayores
# .index nos da los nombres de esas razas

df['breed'] = df['breed'].astype(str).str.strip().str.upper()
df['color'] = df['color'].astype(str).str.strip().str.upper()


top_breeds = df['breed'].value_counts().nlargest(TOP_N).index

print(f"Las razas principales son: {list(top_breeds)}")

# 3. Aplicar la transformación
# Usamos numpy 'where': Si la raza está en 'top_breeds', déjala igual.
# Si NO está, cámbiala por 'Other'.
df['breed_Grouped'] = np.where(df['breed'].isin(top_breeds), df['breed'], 'other')

# 4. Ahora sí, aplicamos One-Hot Encoding a esta nueva columna reducida
df = pd.get_dummies(df, columns=['breed_Grouped'], prefix='breed')

# Verificamos el resultado (ahora tendrás columnas como 'Breed_Labrador', 'Breed_Other', etc.)
print(df.columns)