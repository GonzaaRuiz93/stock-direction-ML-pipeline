
from datetime import datetime

def generar_reporte(metricas_LR, metricas_RF, umbral_LR, umbral_RF, n=0):
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    #if n == 0:
    nombre_archivo = "reporte_modelo.txt"
    #else:
    #    nombre_archivo = f"reporte_modelo_{n}.txt"

    with open(nombre_archivo, "w") as f:
        f.write(f"\n{'='*45}\n")
        #if n != 0:
        f.write(f"REPORTE DE RENDIMIENTO - {timestamp}\n")
        #else:
        #    f.write(f"REPORTE DE RENDIMIENTO V.{n} - {timestamp}\n")
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
        escribir_bloque("METRICAS: Logistic Regression", umbral_LR, metricas_LR)
        escribir_bloque("METRICAS: Random Forest", umbral_RF, metricas_RF)
        
    print(f"📄 Reporte generado en: {nombre_archivo}")