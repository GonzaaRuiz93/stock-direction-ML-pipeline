
from data import Get_process_data, Get_target
from models import Get_ML_Model
from datetime import datetime, timedelta
from gen_reporte import generar_reporte
from visualization import Get_comparative_figure

print("Iniciando Programa")

#Definir variables financieras
TICKER = "AAPL"
UMBRAL_LR = 0.45
UMBRAL_RF = 0.55


#Definir fechas
fecha_actual = datetime.now()
fecha_anterior = (fecha_actual - timedelta(days=15)).strftime('%Y-%m-%d')
fecha_final = (fecha_actual + timedelta(days=1)).strftime('%Y-%m-%d')


#Definir x
X = Get_process_data(TICKER, "2023-01-01", "2026-01-01")
#Usar X para calcular y. Cada fila debe incluir su objetivo para el dia siguiente
y = Get_target(X)

#Quitar la ultima fila de ambos df ya que la ultima fila de y es NaN, por tanto no se puede usar
X=X.drop(X.tail(1).index)
y=y.drop(y.tail(1).index)


#Crear Modelos de ML
LR, RF, metricas_LR, metricas_RF = Get_ML_Model(X, y, UMBRAL_LR, UMBRAL_RF)


# Mostrar Metricas
Get_comparative_figure(metricas_LR, metricas_RF, "Métricas: Solo Logistic Regresion", "Métricas: Modelo Completo (LR + RF)")

#Generar reporte
generar_reporte(metricas_LR, metricas_RF, UMBRAL_LR, UMBRAL_RF)


print("Programa Terminado")

