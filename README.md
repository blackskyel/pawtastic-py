# API for pet adoption probability prediction in days

This is a non-profit app for our tesis.

## Dataset
It is a subset of "Louisville Metro KY - Animal Service Intake and Outcome"[https://catalog.data.gov/dataset/louisville-metro-ky-animal-service-intake-and-outcome]

## Porpouse
Obtain the probable amount of days that a pet will remain in a shelter before being adopted based on a few variables.

## Model
pet_adoption_multimodel.pkl

1. REGRESOR (DÍAS EXACTOS)
   - MAE (Error Promedio): +/- 18.51 días

2. CLASIFICADORES (PROBABILIDADES)
   - Probabilidad < 30 días: Accuracy 89.7%
   - Probabilidad < 60 días: Accuracy 94.7%
   - Probabilidad < 90 días: Accuracy 95.6%
   - Probabilidad < 120 días: Accuracy 96.0%
   - Probabilidad < 150 días: Accuracy 96.1%

3. VARIABLES TOP SELECCIONADAS (DICCIONARIO)
   - Razas Top (12): 
     [DOMESTIC SH, PIT BULL, LABRADOR RETR, GERM SHEPHERD, DOMESTIC MH, BEAGLE, BOXER, DOMESTIC LH, CHIHUAHUA SH, SHIH TZU, SIBERIAN HUSKY, ALASKAN HUSKY]
     
   - Colores Top (10): 
     [BLACK, TABBY, WHITE, BROWN, GRAY, TAN, BRINDLE, TORTIE, ORANGE, CALICO]

## API

### Framework
Evaluación entre Flask y FastAPI

| Característica | Flask | FastAPI | Veredicto |
| :--- | :--- | :--- | :--- |
| **Velocidad** | Lento (Sincrónico por defecto) | **Muy Rápido** (Asíncrono, sobre Starlette) | FastAPI gana 🚀 |
| **Validación de Datos** | Manual (requiere librerías extra) | **Automática** (Nativa con Pydantic) | FastAPI gana 🛡️ |
| **Documentación** | Manual | **Automática** (Genera Swagger UI solo) | FastAPI gana 📄 |
| **Tipado** | Dinámico (Python clásico) | Estricto (Type Hints) | FastAPI gana 👨‍💻 |
| **Uso en ML** | Común por antigüedad | **Estándar moderno** por validación de tipos | FastAPI es el líder |

