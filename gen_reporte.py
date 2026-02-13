
from datetime import datetime

def generar_reporte(metricas_LR, metricas_RF, umbral_LR, umbral_RF, n=0):
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    nombre_archivo = "reporte_modelo.txt"

    with open(nombre_archivo, "w") as f:
        f.write(f"\n{'='*45}\n")
        f.write(f"PERFORMANCE REPORT - {timestamp}\n")
        f.write(f"{'='*45}\n\n")

        # Función interna para escribir cada bloque
        def escribir_bloque(titulo, umbral, datos):
            f.write(f"--- {titulo} ---\n")
            f.write(f"Umbral: {umbral}\n\n")
            for clave, valor in datos.items():
                formato = ".4f" if valor <= 1 else ".2f"
                f.write(f"{clave:15}: {valor:{formato}}\n")
            f.write("\n")

        #Generacion de las MÉTRICAS
        escribir_bloque("METRICS: Logistic Regression", umbral_LR, metricas_LR)
        escribir_bloque("METRICS: Random Forest", umbral_RF, metricas_RF)
        
    print(f"Report generated in: {nombre_archivo}")