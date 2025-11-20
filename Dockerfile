# Usamos una imagen base oficial de Python (ligera)
FROM python:3.10-slim

# Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# 1. Copiamos los requisitos e instalamos dependencias
# Esto se hace antes de copiar el código para aprovechar la caché de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copiamos las carpetas de tu proyecto
# Copiamos la carpeta 'api' local a '/app/api' en el contenedor
COPY api ./api
# Copiamos la carpeta 'model' local a '/app/model' en el contenedor
COPY model ./model

# 3. Exponemos el puerto donde corre FastAPI
EXPOSE 8000

# 4. Comando de arranque
# Ejecutamos uvicorn desde la raíz (/app)
# 'api.main:app' significa: carpeta api -> archivo main.py -> objeto app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]